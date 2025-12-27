"""
NSF Merit Review Criteria
Hardcoded criteria based on NSF's two core merit review principles.
Reference: NSF Proposal & Award Policies & Procedures Guide (PAPPG)
"""

from typing import List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluator import RubricCriterion


def get_nsf_merit_criteria() -> List[RubricCriterion]:
    """
    Returns the two NSF merit review criteria as RubricCriterion objects.

    NSF evaluates all proposals against two criteria:
    1. Intellectual Merit - potential to advance knowledge
    2. Broader Impacts - potential to benefit society

    Both criteria are evaluated across 5 dimensions:
    - Advancement potential
    - Creativity/originality
    - Plan quality and assessment
    - Team qualifications
    - Resource adequacy

    Returns:
        List of 2 RubricCriterion objects with equal weighting (50 points each)
    """

    criteria = [
        RubricCriterion(
            id="nsf_intellectual_merit",
            title="Intellectual Merit",
            description=(
                "The Intellectual Merit criterion encompasses the potential to advance knowledge. "
                "Reviewers evaluate: (1) the potential for the proposed activity to advance knowledge "
                "and understanding within its own field or across different fields; (2) the extent to which "
                "the proposed activities suggest and explore creative, original, or potentially transformative "
                "concepts; (3) whether the plan for carrying out the proposed activities is well-reasoned, "
                "well-organized, and based on a sound rationale with mechanisms to assess success; "
                "(4) the qualifications of the individual, team, or organization to conduct the proposed "
                "activities; and (5) the availability of adequate resources (at the home organization or "
                "through collaborations) to carry out the proposed activities."
            ),
            weight_points=50,
            keywords=[
                "advance knowledge",
                "research contribution",
                "innovation",
                "theoretical framework",
                "methodology",
                "research design",
                "intellectual contribution",
                "novel approach",
                "scientific significance",
                "transformative",
                "creative",
                "original",
                "research questions",
                "hypotheses",
                "expected outcomes",
                "project plan",
                "timeline",
                "milestones",
                "assessment",
                "evaluation metrics",
                "PI qualifications",
                "team expertise",
                "prior work",
                "publications",
                "facilities",
                "equipment",
                "resources",
                "collaborations"
            ],
            compliance_requirements=[]
        ),

        RubricCriterion(
            id="nsf_broader_impacts",
            title="Broader Impacts",
            description=(
                "The Broader Impacts criterion encompasses the potential to benefit society and contribute "
                "to the achievement of specific, desired societal outcomes. Reviewers evaluate: (1) the "
                "potential for the proposed activity to benefit society or advance desired societal outcomes; "
                "(2) the extent to which the proposed activities suggest and explore creative, original, or "
                "potentially transformative approaches to broader impacts; (3) whether the plan for the broader "
                "impacts activities is well-reasoned, well-organized, and based on a sound rationale with "
                "mechanisms to assess success; (4) the qualifications of the individual, team, or organization "
                "to conduct the proposed broader impacts activities; and (5) the availability of adequate "
                "resources to carry out the proposed broader impacts activities."
            ),
            weight_points=50,
            keywords=[
                "societal benefit",
                "broader impacts",
                "education",
                "outreach",
                "diversity",
                "inclusion",
                "underrepresented groups",
                "K-12",
                "undergraduate",
                "graduate students",
                "workforce development",
                "public engagement",
                "science communication",
                "technology transfer",
                "economic impact",
                "policy implications",
                "environmental sustainability",
                "national security",
                "public health",
                "dissemination",
                "open science",
                "data sharing",
                "community partnerships",
                "societal outcomes",
                "broader impacts plan",
                "evaluation strategy",
                "assessment metrics",
                "outreach activities"
            ],
            compliance_requirements=[]
        )
    ]

    return criteria


def validate_nsf_criteria() -> bool:
    """
    Validates that NSF criteria are properly structured.

    Returns:
        True if valid, raises ValueError if not
    """
    criteria = get_nsf_merit_criteria()

    if len(criteria) != 2:
        raise ValueError(f"Expected 2 NSF criteria, got {len(criteria)}")

    total_weight = sum(c.weight_points for c in criteria)
    if total_weight != 100:
        raise ValueError(f"NSF criteria weights must sum to 100, got {total_weight}")

    for criterion in criteria:
        if not criterion.id or not criterion.title or not criterion.description:
            raise ValueError(f"Criterion {criterion.id} missing required fields")
        if criterion.weight_points <= 0:
            raise ValueError(f"Criterion {criterion.id} has invalid weight: {criterion.weight_points}")
        if not criterion.keywords:
            raise ValueError(f"Criterion {criterion.id} has no keywords")

    return True


if __name__ == "__main__":
    # Test the criteria
    criteria = get_nsf_merit_criteria()
    validate_nsf_criteria()

    print("NSF Merit Review Criteria")
    print("=" * 80)
    for i, criterion in enumerate(criteria, 1):
        print(f"\n{i}. {criterion.title} (Weight: {criterion.weight_points} points)")
        print(f"   ID: {criterion.id}")
        print(f"   Description: {criterion.description[:100]}...")
        print(f"   Keywords: {len(criterion.keywords)} keywords")
        print(f"   Top keywords: {', '.join(criterion.keywords[:5])}")

    print(f"\n{'=' * 80}")
    print(f"Total weight: {sum(c.weight_points for c in criteria)} points")
    print("Validation: PASSED")
