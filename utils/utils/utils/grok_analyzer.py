from groq import Groq
import os
import json
import re


def get_groq_client(api_key: str = None) -> Groq:
    """Groq client banao."""
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY nahi mili!")
    return Groq(api_key=key)


def analyze_resume_with_grok(
    resume_text: str,
    jd_text: str,
    ats_scores: dict,
    api_key: str = None
) -> dict:
    client = get_groq_client(api_key)

    breakdown = ats_scores.get("breakdown", {})
    keyword_data = breakdown.get("keyword_match", {})
    section_data = breakdown.get("sections", {})
    format_data  = breakdown.get("formatting", {})
    action_data  = breakdown.get("action_verbs", {})

    prompt = f"""You are an expert ATS resume analyzer used by top companies like Google, Microsoft, Amazon.

RESUME TEXT:
{resume_text[:3000]}

JOB DESCRIPTION:
{jd_text[:1500] if jd_text else "No job description provided - do general analysis"}

CURRENT ATS SCORES:
- Overall Score: {ats_scores.get('final_score', 0)}/100
- Keyword Match: {keyword_data.get('score', 0)}/100
- Missing Keywords: {keyword_data.get('missing_keywords', [])[:10]}
- Missing Sections: {section_data.get('missing_sections', [])}
- Formatting Issues: {format_data.get('issues', [])}
- Action Verbs Count: {action_data.get('action_verbs_count', 0)}
- Quantified Achievements: {action_data.get('quantified_achievements', 0)}

Respond ONLY in this JSON format, no extra text:
{{
    "overall_assessment": "2-3 sentence overall assessment",
    "top_strengths": ["strength 1", "strength 2", "strength 3"],
    "critical_improvements": [
        {{
            "issue": "issue title",
            "explanation": "why this is important",
            "fix": "exactly how to fix this"
        }}
    ],
    "keyword_suggestions": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
    "bullet_point_rewrites": [
        {{
            "original": "weak bullet from resume",
            "improved": "stronger version with action verb + quantification"
        }}
    ],
    "section_suggestions": "advice on missing or weak sections",
    "ats_tips": ["tip1", "tip2", "tip3"],
    "final_verdict": "Will this resume pass ATS? Why or why not in 2 sentences."
}}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert ATS resume analyzer. Always respond with valid JSON only, no markdown, no extra text."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=2000,
            temperature=0.3,
        )

        raw_response = response.choices[0].message.content.strip()
        raw_response = re.sub(r'```json\n?', '', raw_response)
        raw_response = re.sub(r'```\n?', '', raw_response)

        result = json.loads(raw_response)
        return {"success": True, "analysis": result}

    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"JSON parse error: {str(e)}",
            "raw": raw_response if 'raw_response' in locals() else ""
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Groq API error: {str(e)}"
        }


def get_quick_tips(resume_text: str, api_key: str = None) -> str:
    client = get_groq_client(api_key)
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": f"Give exactly 5 quick actionable ATS improvement tips for this resume. Be specific. Resume: {resume_text[:2000]}"
            }],
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Tips load nahi ho sake: {str(e)}"
