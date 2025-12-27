"""
Rubric Utilities
Tools for validating, parsing, and managing custom evaluation rubrics.
"""

import json
from typing import List, Dict, Any, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluator import RubricCriterion


class RubricValidationError(Exception):
    """Raised when a custom rubric fails validation."""
    pass


def validate_rubric_json(rubric_data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validates a custom rubric JSON structure.

    Expected schema:
    {
        "rubric_name": "My Custom Rubric",
        "rubric_description": "Optional description",
        "criteria": [
            {
                "id": "criterion_1",
                "title": "Criterion Title",
                "description": "What evaluators look for",
                "weight_points": 20,
                "keywords": ["keyword1", "keyword2"],
                "compliance_requirements": []  # Optional
            }
        ]
    }

    Args:
        rubric_data: Parsed JSON dictionary

    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """

    # Check top-level structure
    if not isinstance(rubric_data, dict):
        return False, "Rubric must be a JSON object"

    if "criteria" not in rubric_data:
        return False, "Missing required field: 'criteria'"

    criteria = rubric_data["criteria"]
    if not isinstance(criteria, list):
        return False, "'criteria' must be an array"

    if len(criteria) == 0:
        return False, "Rubric must contain at least 1 criterion"

    if len(criteria) > 20:
        return False, f"Too many criteria ({len(criteria)}). Maximum is 20."

    # Validate each criterion
    criterion_ids = set()
    total_weight = 0

    for i, criterion in enumerate(criteria):
        prefix = f"Criterion {i + 1}"

        # Check required fields
        required_fields = ["id", "title", "description", "weight_points", "keywords"]
        for field in required_fields:
            if field not in criterion:
                return False, f"{prefix}: Missing required field '{field}'"

        # Validate id
        criterion_id = criterion["id"]
        if not isinstance(criterion_id, str) or not criterion_id.strip():
            return False, f"{prefix}: 'id' must be a non-empty string"
        if criterion_id in criterion_ids:
            return False, f"{prefix}: Duplicate criterion ID '{criterion_id}'"
        criterion_ids.add(criterion_id)

        # Validate title
        if not isinstance(criterion["title"], str) or not criterion["title"].strip():
            return False, f"{prefix}: 'title' must be a non-empty string"

        # Validate description
        if not isinstance(criterion["description"], str) or not criterion["description"].strip():
            return False, f"{prefix}: 'description' must be a non-empty string"

        # Validate weight_points
        weight = criterion["weight_points"]
        if not isinstance(weight, (int, float)):
            return False, f"{prefix}: 'weight_points' must be a number"
        if weight <= 0:
            return False, f"{prefix}: 'weight_points' must be greater than 0"
        if weight > 100:
            return False, f"{prefix}: 'weight_points' cannot exceed 100"
        total_weight += weight

        # Validate keywords
        keywords = criterion["keywords"]
        if not isinstance(keywords, list):
            return False, f"{prefix}: 'keywords' must be an array"
        if len(keywords) == 0:
            return False, f"{prefix}: Must provide at least 1 keyword"
        for j, keyword in enumerate(keywords):
            if not isinstance(keyword, str) or not keyword.strip():
                return False, f"{prefix}: Keyword {j + 1} must be a non-empty string"

        # Validate compliance_requirements (optional)
        if "compliance_requirements" in criterion:
            reqs = criterion["compliance_requirements"]
            if not isinstance(reqs, list):
                return False, f"{prefix}: 'compliance_requirements' must be an array"
            for j, req in enumerate(reqs):
                if not isinstance(req, str):
                    return False, f"{prefix}: Compliance requirement {j + 1} must be a string"

    # Check total weight
    if total_weight != 100:
        return False, f"Total weight must equal 100 points, got {total_weight}"

    return True, "Validation successful"


def parse_rubric_json(rubric_data: Dict[str, Any]) -> List[RubricCriterion]:
    """
    Parses a validated rubric JSON into RubricCriterion objects.

    Args:
        rubric_data: Validated JSON dictionary

    Returns:
        List of RubricCriterion objects

    Raises:
        RubricValidationError: If validation fails
    """

    is_valid, error_msg = validate_rubric_json(rubric_data)
    if not is_valid:
        raise RubricValidationError(error_msg)

    criteria_list = []
    for criterion_data in rubric_data["criteria"]:
        criterion = RubricCriterion(
            id=criterion_data["id"],
            title=criterion_data["title"],
            description=criterion_data["description"],
            weight_points=int(criterion_data["weight_points"]),
            keywords=criterion_data["keywords"],
            compliance_requirements=criterion_data.get("compliance_requirements", [])
        )
        criteria_list.append(criterion)

    return criteria_list


def load_rubric_from_file(file_path: str) -> List[RubricCriterion]:
    """
    Loads and parses a custom rubric from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        List of RubricCriterion objects

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file isn't valid JSON
        RubricValidationError: If rubric structure is invalid
    """

    with open(file_path, 'r', encoding='utf-8') as f:
        rubric_data = json.load(f)

    return parse_rubric_json(rubric_data)


def get_example_rubric() -> Dict[str, Any]:
    """
    Returns an example custom rubric JSON structure.

    This can be used as a template for users creating custom rubrics.

    Returns:
        Dictionary representing a valid custom rubric
    """

    return {
        "rubric_name": "Example Grant Evaluation Rubric",
        "rubric_description": "A sample rubric for evaluating research grant proposals",
        "criteria": [
            {
                "id": "project_significance",
                "title": "Project Significance and Impact",
                "description": (
                    "Evaluates the potential significance and real-world impact of the proposed project. "
                    "Strong proposals clearly articulate the problem being addressed, demonstrate its "
                    "importance to the field and society, and explain how the project will make a meaningful "
                    "contribution. Look for evidence of need, potential beneficiaries, and expected outcomes."
                ),
                "weight_points": 25,
                "keywords": [
                    "significance",
                    "impact",
                    "importance",
                    "contribution",
                    "problem statement",
                    "need",
                    "beneficiaries",
                    "outcomes",
                    "real-world application"
                ],
                "compliance_requirements": []
            },
            {
                "id": "innovation_approach",
                "title": "Innovation and Approach",
                "description": (
                    "Assesses the novelty and creativity of the proposed approach. Strong proposals demonstrate "
                    "innovative thinking, whether through new methodologies, technologies, or applications of "
                    "existing approaches to new contexts. The approach should be well-justified and "
                    "appropriately rigorous for the project goals."
                ),
                "weight_points": 25,
                "keywords": [
                    "innovation",
                    "novel",
                    "creative",
                    "methodology",
                    "approach",
                    "research design",
                    "methods",
                    "technology",
                    "rigor"
                ],
                "compliance_requirements": []
            },
            {
                "id": "team_qualifications",
                "title": "Team Qualifications and Capacity",
                "description": (
                    "Evaluates whether the project team has the necessary expertise, experience, and resources "
                    "to successfully execute the proposed work. Strong proposals highlight relevant "
                    "qualifications, prior accomplishments, complementary skills across team members, and "
                    "access to required facilities or partnerships."
                ),
                "weight_points": 20,
                "keywords": [
                    "qualifications",
                    "expertise",
                    "experience",
                    "team",
                    "personnel",
                    "capacity",
                    "prior work",
                    "track record",
                    "facilities",
                    "partnerships"
                ],
                "compliance_requirements": []
            },
            {
                "id": "implementation_plan",
                "title": "Implementation Plan and Feasibility",
                "description": (
                    "Assesses the quality and feasibility of the implementation plan. Strong proposals include "
                    "clear timelines, realistic milestones, well-defined roles and responsibilities, and "
                    "evidence-based strategies. The plan should demonstrate that the project is achievable "
                    "within the proposed timeframe and budget."
                ),
                "weight_points": 20,
                "keywords": [
                    "implementation",
                    "plan",
                    "timeline",
                    "milestones",
                    "feasibility",
                    "schedule",
                    "deliverables",
                    "work plan",
                    "project management"
                ],
                "compliance_requirements": []
            },
            {
                "id": "evaluation_sustainability",
                "title": "Evaluation and Sustainability",
                "description": (
                    "Evaluates the project's evaluation plan and sustainability strategy. Strong proposals "
                    "include clear metrics for success, rigorous evaluation methodologies, and realistic "
                    "plans for sustaining project outcomes beyond the funding period. Look for evidence of "
                    "thoughtful measurement approaches and long-term vision."
                ),
                "weight_points": 10,
                "keywords": [
                    "evaluation",
                    "assessment",
                    "metrics",
                    "sustainability",
                    "long-term",
                    "outcomes measurement",
                    "success indicators",
                    "continuation plan"
                ],
                "compliance_requirements": []
            }
        ]
    }


def save_example_rubric(output_path: str) -> None:
    """
    Saves the example rubric to a JSON file.

    Args:
        output_path: Where to save the example rubric
    """

    example = get_example_rubric()
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(example, f, indent=2)

    print(f"Example rubric saved to: {output_path}")


if __name__ == "__main__":
    # Test the validator with example rubric
    example = get_example_rubric()

    print("Validating example rubric...")
    is_valid, msg = validate_rubric_json(example)

    if is_valid:
        print(f"Validation: PASSED - {msg}")

        print("\nParsing rubric...")
        criteria = parse_rubric_json(example)

        print(f"\nParsed {len(criteria)} criteria:")
        for i, criterion in enumerate(criteria, 1):
            print(f"  {i}. {criterion.title} ({criterion.weight_points} points)")
            print(f"     Keywords: {len(criterion.keywords)} items")

        print(f"\nTotal weight: {sum(c.weight_points for c in criteria)} points")

        # Save example to file
        output_file = "src/data/example_custom_rubric.json"
        save_example_rubric(output_file)

    else:
        print(f"Validation: FAILED - {msg}")
