"""Load and normalise NASA ASRS reports.

Source: the `elihoole/asrs-aviation-reports` mirror of public NASA Aviation Safety
Reporting System records (de-identified voluntary reports). We keep the fields an
investigator would use, and split them into two groups on purpose:

  * context (aircraft, phase, FAR part, year ...) -> may appear in the index text
  * labels  (anomaly, primary problem, factors)   -> NEVER indexed; used only to
                evaluate whether retrieved precedents are meaningful.
"""

from __future__ import annotations


COL = {
    "id": "acn_num_ACN",
    "date": "Time_Date",
    "aircraft": "Aircraft 1.2_Make Model Name",
    "operator": "Aircraft 1.1_Aircraft Operator",
    "far_part": "Aircraft 1.5_Operating Under FAR Part",
    "mission": "Aircraft 1.7_Mission",
    "phase": "Aircraft 1.9_Flight Phase",
    "conditions": "Environment_Flight Conditions",
    "anomaly": "Events_Anomaly",
    "contributing": "Assessments_Contributing Factors / Situations",
    "primary_problem": "Assessments.1_Primary Problem",
    "human_factors": "Person 1.7_Human Factors",
    "result": "Events.5_Result",
    "detector": "Events.3_Detector",
    "synopsis": "Report 1.2_Synopsis",
    "narrative": "Report 1_Narrative",
    "narrative2": "Report 2_Narrative",
}


CONTEXT_FIELDS = ("aircraft", "operator", "far_part", "mission", "phase", "conditions")


LIST_FIELDS = ("anomaly", "contributing", "human_factors", "result")



def _s(v) -> str:
    return " ".join(str(v or "").split())



def _split_list(v) -> list[str]:
    return [x.strip() for x in _s(v).split(";") if x.strip()]
