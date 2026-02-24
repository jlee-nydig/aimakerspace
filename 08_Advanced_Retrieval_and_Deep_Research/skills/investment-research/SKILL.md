---
name: investment-research
description: Research investment opportunities and create comparison reports
version: 1.0.0
tools:
  - read_file
  - write_file
  - web_search
---

# Investment Research Skill

You are conducting investment research and creating a comparison report. Follow these steps:

## Step 1: Understand Requirements
Gather information about:
- Investment categories of interest (reinsurance, private equity, real estate, commodities)
- Risk tolerance and return expectations
- Investment horizon and liquidity needs
- Minimum investment thresholds
- Regulatory and compliance considerations

## Step 2: Check Existing Profile
Read the client's portfolio profile if available:
```
workspace/portfolio_assessment.md
```

## Step 3: Research Opportunities
Research investment opportunities including:
- Reinsurance strategies (3-5 options)
- Private equity approaches (3-5 options)
- Real estate vehicles (5-7 options)
- Commodities exposure methods (3-5 options)
- Risk-return profiles for each

## Step 4: Generate Comparison Report
Create a consolidated comparison report organized by:
- Asset Class Overview
- Risk-Return Characteristics
- Liquidity Profiles
- Fee Structures
- Minimum Investment Requirements

## Step 5: Save Outputs
Write the research report to `workspace/investment_research.md` and comparison summary to `workspace/investment_comparison.md`.

## Investment Guidelines
- Balance risk across alternative asset classes
- Consider correlation benefits with traditional portfolio
- Prioritize strategies with transparent fee structures
- Factor in liquidity constraints and lock-up periods
- Suggest implementation timeline for gradual allocation
