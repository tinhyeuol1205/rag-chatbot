from __future__ import annotations

"""
Test Dataset — Bộ câu hỏi + ground truth để đánh giá RAG.

Mỗi sample gồm:
  - question: Câu hỏi thực tế user sẽ hỏi
  - ground_truth: Câu trả lời đúng (lấy từ tài liệu gốc)

Bộ dataset này dùng để:
  1. Chạy câu hỏi qua RAG pipeline → thu được answer + retrieved context
  2. So sánh answer vs ground_truth → tính Faithfulness, Answer Relevance
  3. Đánh giá retrieved context → tính Context Relevance
"""

# Bộ test set — xây dựng từ 2 sample docs (company_policy.md + engineering_handbook.md)
EVAL_DATASET = [
    # ----- Company Policy -----
    {
        "question": "How many days of annual leave do employees get?",
        "ground_truth": "Full-time employees receive 15 days of paid annual leave per year, accrued at 1.25 days per month. Unused leave can be carried over up to a maximum of 5 days to the next calendar year.",
    },
    {
        "question": "What equipment does the company provide for remote workers?",
        "ground_truth": "The company provides a laptop (Dell XPS or MacBook Pro), an external monitor, and a headset for remote work. Employees need a stable internet connection with minimum 50 Mbps download speed.",
    },
    {
        "question": "What is the password policy?",
        "ground_truth": "All passwords must be at least 12 characters long and include uppercase letters, lowercase letters, numbers, and special characters. Passwords must be changed every 90 days and cannot repeat the last 5 passwords. Multi-factor authentication is mandatory.",
    },
    {
        "question": "How long is the parental leave?",
        "ground_truth": "New parents (both maternity and paternity) are eligible for 6 months of fully paid parental leave. This applies to both biological and adoptive parents. Employees must notify HR at least 60 days before the expected leave start date.",
    },
    {
        "question": "What happens during the first day of onboarding?",
        "ground_truth": "New employees receive their laptop and security badge at reception by 9:00 AM. An IT orientation covers email setup, VPN access, Slack workspace, Jira project boards, and the internal knowledge base. Lunch is provided with the onboarding cohort.",
    },
    {
        "question": "What is the daily meal allowance for international business travel?",
        "ground_truth": "Daily meal allowances during business travel are $50 for domestic trips and $80 for international trips. Team dinners require prior manager approval and are capped at $75 per person. Alcohol is not reimbursable.",
    },
    {
        "question": "How should security incidents be reported?",
        "ground_truth": "Any suspected security breach must be reported to the Security Operations Center within 1 hour of discovery. Reports can be submitted via email to soc@techcorp.com or by calling the 24/7 hotline at +1-800-SEC-HELP.",
    },
    # ----- Engineering Handbook -----
    {
        "question": "What is the Git branching strategy?",
        "ground_truth": "TechCorp follows Git Flow with main branches: main (production-ready), develop (integration), feature/* (individual features from develop), and hotfix/* (emergency fixes from main). Branch naming: feature/JIRA-123-short-description.",
    },
    {
        "question": "How many code review approvals are needed before merging?",
        "ground_truth": "All code changes require at least 2 approving reviews before merging. Reviewers should check for correctness, test coverage (minimum 80%), coding standards, and security vulnerabilities. Reviews should be completed within 24 business hours.",
    },
    {
        "question": "What is the incident severity classification?",
        "ground_truth": "SEV1 (Critical): service completely down, 15 min response. SEV2 (Major): significant degradation, 30 min response. SEV3 (Minor): limited impact with workaround, 4 hour response. Post-incident reviews are mandatory for SEV1 and SEV2 within 48 hours.",
    },
]
