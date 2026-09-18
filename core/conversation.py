"""
Conversation orchestration and multi-turn memory.

A `Session` holds the accumulated environmental state across turns, the list of
questions already asked (so the system never repeats one), and the turn history.

Memory behaviour worth noting:
  * Observations accumulate. A value given in turn 1 is still in force in turn 6.
  * A new value for a known metric *overwrites* the old one and is recorded as a
    correction, so "actually the pH is 6.8" revises the diagnosis rather than
    conflicting with it.
  * Answers to a pending question are attributed to the metric that was asked
    about, so a bare reply like "about 450" is interpreted correctly.
  * Assumed values are marked as such and are the first thing the system offers
    to confirm.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.clarify import (
    MIN_METRICS_TO_REASON,
    TARGET_METRICS,
    coverage_note,
    needs_clarification,
    next_questions,
)
from core.evidence import attach_evidence
from core.extract import extract
from core.reasoning import reason
from core.reference import classify, interpretation_for
from core.schema import (
    ClarifyingQuestion,
    EnvState,
    Metric,
    Observation,
    SystemResponse,
)

NUM_ONLY = re.compile(r"^\s*(?:about|around|approx\.?|roughly|~)?\s*(-?\d+(?:\.\d+)?)\s*([a-z%/°]*)\s*$", re.I)
SHORT_WORD = re.compile(r"^[a-z\s\-]{2,30}$", re.I)

RESET_PHRASES = {"reset", "start over", "new site", "clear", "restart"}

# A bare number is only accepted as an answer to the pending question if it is
# plausible for that metric. Otherwise the user is answering something else and
# the parser should stay silent rather than record nonsense such as
# "0.3 habitat types".
TERSE_RANGES = {
    Metric.SOIL_PH: (2.5, 11.0),
    Metric.SOIL_ORGANIC_CARBON: (0.0, 25.0),
    Metric.SOIL_MOISTURE: (0.0, 100.0),
    Metric.TEMPERATURE: (-25.0, 55.0),
    Metric.RAINFALL: (5.0, 10000.0),
    Metric.SPECIES_RICHNESS: (1.0, 100000.0),
    Metric.HABITAT_DIVERSITY: (1.0, 500.0),
    Metric.DEFORESTATION: (0.0, 100.0),
}


def _terse_value_plausible(metric: Metric, value: float) -> bool:
    lo, hi = TERSE_RANGES.get(metric, (-1e12, 1e12))
    if not (lo <= value <= hi):
        return False
    # Counts must be whole numbers.
    if metric in (Metric.HABITAT_DIVERSITY, Metric.SPECIES_RICHNESS) and abs(value - round(value)) > 1e-6:
        return False
    return True


@dataclass
class Turn:
    user_input: str
    response: SystemResponse
    new_observations: List[str] = field(default_factory=list)
    corrections: List[str] = field(default_factory=list)


@dataclass
class Session:
    state: EnvState = field(default_factory=EnvState)
    history: List[Turn] = field(default_factory=list)
    pending_questions: List[ClarifyingQuestion] = field(default_factory=list)
    asked_metrics: List[Metric] = field(default_factory=list)
    force_recommend: bool = False

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.state = EnvState()
        self.history = []
        self.pending_questions = []
        self.asked_metrics = []
        self.force_recommend = False

    # ------------------------------------------------------------------
    def _merge(self, observations: List[Observation]) -> tuple[List[str], List[str]]:
        new, corrected = [], []
        for obs in observations:
            existing = self.state.get(obs.metric)
            label = obs.metric.value.replace("_", " ")
            if existing is None:
                self.state.set(obs)
                new.append(f"{label} = {obs.display()}")
            elif existing.display() != obs.display():
                # A supplied value always beats an assumed one; otherwise the
                # newer statement wins and is logged as a correction.
                if existing.source == "default_assumption" and obs.source != "default_assumption":
                    self.state.set(obs)
                    new.append(f"{label} = {obs.display()} (confirmed, replacing an assumption)")
                elif obs.source == "default_assumption":
                    continue
                else:
                    self.state.set(obs)
                    corrected.append(f"{label}: {existing.display()} -> {obs.display()}")
        return new, corrected

    # ------------------------------------------------------------------
    def _answer_to_pending(self, text: str) -> List[Observation]:
        """Interpret a terse reply as an answer to the question just asked."""
        if not self.pending_questions:
            return []
        stripped = text.strip()
        if len(stripped) > 60 or not stripped:
            return []

        target = self.pending_questions[0].metric
        m = NUM_ONLY.match(stripped)
        if m:
            try:
                value = float(m.group(1))
            except ValueError:
                return []
            unit = m.group(2) or None
            if not _terse_value_plausible(target, value):
                return []
            obs = Observation(
                metric=target, raw=stripped, value=value, unit=unit, source="user"
            )
            return [classify(obs)]

        if SHORT_WORD.match(stripped):
            obs = classify(Observation(metric=target, raw=stripped, category=stripped.lower(), source="user"))
            if obs.classification:
                return [obs]
        return []

    # ------------------------------------------------------------------
    def interpretations(self) -> List[str]:
        out = []
        for metric in self.state.known_core_metrics():
            obs = self.state.get(metric)
            meaning = interpretation_for(metric, obs.classification)
            if meaning:
                label = metric.value.replace("_", " ").title()
                flag = " [assumed]" if obs.source == "default_assumption" else ""
                out.append(f"{label} {obs.display()} -> {obs.classification}{flag}: {meaning}")
        return out

    # ------------------------------------------------------------------
    def process(self, user_input: str, max_questions: int = 2) -> SystemResponse:
        text = (user_input or "").strip()
        self.state.turn += 1

        if text.lower() in RESET_PHRASES:
            self.reset()
            resp = SystemResponse(
                mode="chat", state=self.state,
                message="Session cleared. Describe the site and I will start again.",
            )
            return resp

        # An explicit request to proceed overrides further questioning.
        if re.search(r"\b(just|go ahead|proceed|recommend now|that'?s all|no more questions|don'?t know|not sure|unknown)\b", text.lower()):
            self.force_recommend = True

        observations = extract(text)
        if not observations:
            observations = self._answer_to_pending(text)
        else:
            # Mixed reply: free text plus a bare number answering the question.
            covered = {o.metric for o in observations}
            if self.pending_questions and self.pending_questions[0].metric not in covered:
                observations.extend(self._answer_to_pending(text))

        new, corrected = self._merge(observations)

        known = len(self.state.known_core_metrics())

        # Below the reasoning floor, the system asks rather than guesses. This is
        # the only case that produces questions *instead of* an answer.
        if needs_clarification(self.state) and not self.force_recommend:
            questions = [q for q in next_questions(self.state, max_questions=max_questions)
                         if q.metric not in self.asked_metrics]
            if not questions:
                questions = next_questions(self.state, max_questions=max_questions)
            if questions:
                self.pending_questions = questions
                self.asked_metrics.extend(q.metric for q in questions)
                resp = SystemResponse(
                    mode="clarify",
                    state=self.state,
                    questions=questions,
                    interpretation=self.interpretations(),
                    coverage_note=coverage_note(self.state),
                    message=self._clarify_message(new, corrected, known),
                )
                self.history.append(Turn(text, resp, new, corrected))
                return resp

        # Above the floor, the system answers. It may still attach a follow-up
        # question, because more detail would sharpen the answer - but it does
        # not withhold the analysis to get it.
        follow_up = []
        if (
            known < TARGET_METRICS
            and not self.force_recommend
            and len(self.history) < 8
        ):
            follow_up = [q for q in next_questions(self.state, max_questions=1)
                         if q.metric not in self.asked_metrics]
            self.asked_metrics.extend(q.metric for q in follow_up)

        self.pending_questions = follow_up
        recommendations = reason(self.state)
        recommendations = attach_evidence(recommendations)

        if not recommendations:
            resp = SystemResponse(
                mode="clarify",
                state=self.state,
                questions=next_questions(self.state, max_questions=max_questions),
                interpretation=self.interpretations(),
                coverage_note=coverage_note(self.state),
                message=(
                    "No rule in the reasoning engine matched this combination of variables, so I will not "
                    "invent advice. The variables below would let me evaluate it properly."
                ),
            )
            self.history.append(Turn(text, resp, new, corrected))
            return resp

        resp = SystemResponse(
            mode="recommend",
            state=self.state,
            recommendations=recommendations,
            questions=follow_up,
            interpretation=self.interpretations(),
            coverage_note=coverage_note(self.state),
            message=self._recommend_message(new, corrected, known),
        )
        self.history.append(Turn(text, resp, new, corrected))
        return resp

    # ------------------------------------------------------------------
    def _unasked(self) -> List[Metric]:
        known = set(self.state.known_core_metrics())
        return [m for m in self.state.missing_core_metrics()
                if m not in self.asked_metrics and m not in known]

    def _clarify_message(self, new: List[str], corrected: List[str], known: int) -> str:
        parts = []
        if corrected:
            parts.append("Updated: " + "; ".join(corrected) + ".")
        if new:
            parts.append("Recorded: " + "; ".join(new) + ".")
        if known < MIN_METRICS_TO_REASON:
            parts.append(
                f"I need at least {MIN_METRICS_TO_REASON} environmental variables before I can reason across them "
                "rather than give single-variable advice."
            )
        else:
            parts.append("I can already reason with this, but the following would materially change the answer.")
        return " ".join(parts)

    def _recommend_message(self, new: List[str], corrected: List[str], known: int) -> str:
        parts = []
        if corrected:
            parts.append("Updated: " + "; ".join(corrected) + ", and the analysis below reflects that.")
        if new and not corrected:
            parts.append("Recorded: " + "; ".join(new) + ".")
        return " ".join(parts)
