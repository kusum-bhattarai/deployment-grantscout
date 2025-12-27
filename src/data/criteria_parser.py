"""
Natural Language Criteria Parser
Converts user-provided plain text criteria descriptions into structured RubricCriterion objects.
"""

import json
import os
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluator import RubricCriterion

load_dotenv()


def parse_natural_language_criteria(criteria_text: str) -> List[RubricCriterion]:
    """
    Parse natural language criteria description into RubricCriterion objects using LLM.

    Args:
        criteria_text: Plain text description of evaluation criteria

    Returns:
        List of RubricCriterion objects

    Raises:
        ValueError: If parsing fails or criteria are invalid
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = f"""You are an expert in grant evaluation rubric design. Convert the following natural language description of evaluation criteria into a structured JSON format.

**User's Criteria Description:**
{criteria_text}

**Your Task:**
1. Identify distinct evaluation criteria from the description
2. For each criterion, extract or infer:
   - A unique ID (lowercase with underscores, e.g., "project_impact")
   - A clear title (e.g., "Project Impact and Significance")
   - A detailed description of what evaluators should look for
   - Weight/importance (must sum to 100 across all criteria)
   - Relevant keywords for searching the proposal

**Rules:**
- Create 2-10 criteria (most rubrics have 4-6)
- Weights MUST sum to exactly 100
- Each criterion needs at least 3 keywords
- Descriptions should be detailed and actionable
- If weights aren't specified, distribute evenly or based on implied importance

**Output Format (JSON):**
{{
  "rubric_name": "Descriptive name for this rubric",
  "criteria": [
    {{
      "id": "criterion_id",
      "title": "Criterion Title",
      "description": "Detailed description of what strong proposals demonstrate for this criterion...",
      "weight_points": 25,
      "keywords": ["keyword1", "keyword2", "keyword3", "keyword4"],
      "compliance_requirements": []
    }}
  ]
}}

**Examples of Good Criteria:**
- Project Significance/Impact: How important is the problem? What's the potential benefit?
- Innovation/Novelty: How creative or transformative is the approach?
- Team Qualifications: Does the team have necessary expertise and resources?
- Implementation Plan: Is the plan realistic, detailed, and feasible?
- Sustainability: How will outcomes be sustained beyond the funding period?

Respond ONLY with valid JSON. Ensure weights sum to 100.
"""

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are an expert in grant evaluation rubric design. You convert natural language criteria into structured JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        rubric_json = json.loads(response.choices[0].message.content)

        # Validate the structure
        if "criteria" not in rubric_json:
            raise ValueError("Generated rubric missing 'criteria' field")

        # Convert to RubricCriterion objects
        criteria_list = []
        total_weight = 0

        for crit_data in rubric_json["criteria"]:
            # Validate required fields
            required = ["id", "title", "description", "weight_points", "keywords"]
            missing = [f for f in required if f not in crit_data]
            if missing:
                raise ValueError(f"Criterion missing required fields: {missing}")

            criterion = RubricCriterion(
                id=crit_data["id"],
                title=crit_data["title"],
                description=crit_data["description"],
                weight_points=int(crit_data["weight_points"]),
                keywords=crit_data["keywords"],
                compliance_requirements=crit_data.get("compliance_requirements", [])
            )

            criteria_list.append(criterion)
            total_weight += criterion.weight_points

        # Validate total weight
        if total_weight != 100:
            # Attempt to normalize weights
            adjustment_factor = 100 / total_weight
            for criterion in criteria_list:
                criterion.weight_points = round(criterion.weight_points * adjustment_factor)

            # Final check
            total_weight = sum(c.weight_points for c in criteria_list)
            if total_weight != 100:
                # Adjust the largest criterion to make it exactly 100
                diff = 100 - total_weight
                largest = max(criteria_list, key=lambda c: c.weight_points)
                largest.weight_points += diff

        return criteria_list

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error parsing criteria: {str(e)}")


if __name__ == "__main__":
    # Test the parser
    test_input = """
    I want to evaluate grant proposals based on three main areas:

    1. Scientific Merit (40%): How innovative and rigorous is the research approach?
    2. Team Expertise (30%): Does the team have the right qualifications?
    3. Broader Impact (30%): What's the societal benefit and outreach plan?
    """

    print("Testing Natural Language Criteria Parser")
    print("=" * 60)
    print(f"Input:\n{test_input}\n")

    try:
        criteria = parse_natural_language_criteria(test_input)
        print(f"✓ Parsed {len(criteria)} criteria:\n")

        for i, crit in enumerate(criteria, 1):
            print(f"{i}. {crit.title} ({crit.weight_points} points)")
            print(f"   ID: {crit.id}")
            print(f"   Description: {crit.description[:80]}...")
            print(f"   Keywords ({len(crit.keywords)}): {', '.join(crit.keywords[:5])}")
            print()

        total_weight = sum(c.weight_points for c in criteria)
        print(f"Total weight: {total_weight} points")

        if total_weight == 100:
            print("✅ Weights validated!")
        else:
            print(f"❌ Weight error: sum is {total_weight}, not 100")

    except Exception as e:
        print(f"❌ Error: {e}")
