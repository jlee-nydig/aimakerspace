---
name: portfolio-assessment
description: Assess client investment portfolio and create personalized recommendations
version: 1.0.0
tools:
  - read_file
  - write_file
---

# Portfolio Assessment Skill

You are conducting a comprehensive investment portfolio assessment. Follow these steps:

## Step 1: Gather Information
Ask the client about:
- Current portfolio allocation (equities, bonds, alternatives, cash)
- Investment goals (growth, income, preservation, diversification)
- Risk tolerance and investment horizon
- Regulatory status (accredited investor, qualified purchaser, etc.)
- Any constraints or preferences (ESG, sector exclusions, liquidity needs)
- Current knowledge of alternative investments

## Step 2: Analyze Responses
Review the client's answers and identify:
- Primary investment priority
- Secondary goals
- Potential barriers to implementation (liquidity, minimums, accreditation)
- Existing portfolio strengths to build on

## Step 3: Create Assessment Report
Write a portfolio assessment report to `workspace/portfolio_assessment.md` containing:
- Summary of current portfolio state
- Identified strengths
- Areas for improvement
- Recommended focus areas (prioritized)
- Suggested next steps

## Step 4: Provide Recommendations
Based on the assessment, provide:
- 3 immediate action items (can start this quarter)
- 3 short-term goals (1-2 quarters)
- 3 long-term goals (1-3 years)

## Output Format
Always save results to the workspace directory and provide a clear summary to the client.
