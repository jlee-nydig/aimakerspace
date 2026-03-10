"""Stone Ridge Investment MCP Server — example custom MCP server.

Provides four investment-themed tools with hardcoded data derived from
the Stone Ridge 2025 Investor Letter.

Run with:
    uv run mcp dev mcp_server.py
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Stone Ridge Investment Tools")

# ---------------------------------------------------------------------------
# Hardcoded reference data (sourced from the 2025 Investor Letter)
# ---------------------------------------------------------------------------

FUNDS = {
    "sre": {
        "name": "Stone Ridge Reinsurance Risk Premium Interval Fund (SRE)",
        "overview": (
            "SRE provides investors with access to the reinsurance risk premium — "
            "the return for bearing catastrophe risk such as hurricanes and earthquakes. "
            "Stone Ridge believes reinsurance offers a genuinely uncorrelated return "
            "stream driven by physical events rather than financial-market sentiment."
        ),
        "key_themes": [
            "Catastrophe risk as an asset class",
            "Low correlation to equities and fixed income",
            "Bayesian pricing of natural-disaster frequency and severity",
        ],
    },
    "longtail re": {
        "name": "Longtail Re",
        "overview": (
            "Longtail Re is a Bermuda-based reinsurer wholly owned by Stone Ridge. "
            "It writes excess-of-loss reinsurance contracts across global property "
            "catastrophe risks.  The 2025 letter highlights strong underwriting "
            "discipline and favorable market conditions post-2023 hurricane season."
        ),
        "key_themes": [
            "Bermuda-based reinsurance platform",
            "Excess-of-loss contract specialization",
            "Post-catastrophe market hardening benefits",
        ],
    },
    "bitcoin": {
        "name": "Stone Ridge Bitcoin Strategy",
        "overview": (
            "Stone Ridge allocates a portion of its balance sheet to bitcoin, "
            "viewing it as a long-duration, non-sovereign store of value. The 2025 "
            "letter discusses bitcoin's role in portfolio construction through the "
            "lens of Bayesian probability updating and convexity."
        ),
        "key_themes": [
            "Non-sovereign store of value thesis",
            "Bayesian probability of monetary adoption",
            "Convex payoff profile",
        ],
    },
    "reinsurance": {
        "name": "Stone Ridge Reinsurance Platform",
        "overview": (
            "Stone Ridge's reinsurance platform encompasses both the interval fund (SRE) "
            "and Longtail Re.  The 2025 letter emphasizes that reinsurance premiums are "
            "set by the physical world — wind speeds, fault lines, ocean temperatures — "
            "not by central-bank policy, making the premium a structurally diversifying "
            "return source."
        ),
        "key_themes": [
            "Physically-driven risk premiums",
            "Structural diversification away from financial markets",
            "Long-term positive expected returns from bearing tail risk",
        ],
    },
}

INVESTMENT_PHILOSOPHY = (
    "Stone Ridge's investment philosophy centers on Bayesian thinking — continuously "
    "updating beliefs as new data arrives rather than anchoring to a single forecast. "
    "Key tenets from the 2025 Investor Letter include:\n\n"
    "1. **Diversification across independent risk factors**: True diversification "
    "requires exposure to return streams whose drivers are independent (e.g., "
    "catastrophe risk, energy supply/demand, bitcoin adoption).\n\n"
    "2. **Harvesting risk premiums others avoid**: Stone Ridge targets asset classes "
    "where structural or behavioral biases cause risk premiums to be larger than the "
    "expected loss — reinsurance and longtail catastrophe risk are prime examples.\n\n"
    "3. **Bayesian probability updating**: Rather than making binary bets, Stone Ridge "
    "assigns probabilities to outcomes and updates them with evidence, leading to "
    "measured position sizing.\n\n"
    "4. **Long time horizons and patience**: Many of Stone Ridge's strategies require "
    "a willingness to endure short-term volatility in exchange for long-term "
    "compounding of genuinely uncorrelated returns.\n\n"
    "5. **Radical transparency**: The annual investor letter provides detailed "
    "discussion of mistakes, lessons learned, and evolving views."
)

LETTER_SECTIONS = {
    "overview": (
        "The 2025 Stone Ridge Investor Letter reflects on the past year and outlines "
        "the firm's outlook.  Stone Ridge manages roughly $10 billion across reinsurance, "
        "energy, bitcoin, and alternative lending strategies.  The letter highlights "
        "the firm's commitment to Bayesian reasoning and long-term compounding."
    ),
    "reinsurance": (
        "The reinsurance section discusses market hardening after recent catastrophe "
        "years, improving expected returns.  Stone Ridge emphasizes that reinsurance "
        "premiums compensate investors for physically-driven tail risks — hurricanes, "
        "earthquakes, wildfires — that are independent of equity markets."
    ),
    "bitcoin": (
        "Stone Ridge frames bitcoin as a Bayesian bet: the probability that bitcoin "
        "emerges as a major non-sovereign store of value continues to increase with "
        "each year of network survival and institutional adoption.  Even a modest "
        "probability-weighted allocation can be portfolio-accretive due to the convex "
        "payoff structure."
    ),
    "energy": (
        "The energy section covers Stone Ridge's investments in physical energy assets "
        "and energy-related risk premiums.  The firm notes that global energy demand "
        "is structurally growing due to AI/data-center buildout, supporting long-term "
        "returns from energy infrastructure."
    ),
    "philosophy": INVESTMENT_PHILOSOPHY,
}


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def get_fund_overview(fund_name: str) -> str:
    """Return an overview of a Stone Ridge fund.

    Args:
        fund_name: Name or keyword for the fund — one of 'SRE', 'Longtail Re',
                   'Bitcoin', or 'Reinsurance'.
    """
    key = fund_name.strip().lower()
    fund = FUNDS.get(key)
    if fund is None:
        # fuzzy match
        for k, v in FUNDS.items():
            if key in k or key in v["name"].lower():
                fund = v
                break
    if fund is None:
        return (
            f"Fund '{fund_name}' not found. Available funds: "
            + ", ".join(f["name"] for f in FUNDS.values())
        )

    themes = "\n".join(f"  - {t}" for t in fund["key_themes"])
    return f"**{fund['name']}**\n\n{fund['overview']}\n\nKey themes:\n{themes}"


@mcp.tool()
def get_investment_philosophy() -> str:
    """Return Stone Ridge's investment philosophy as described in the 2025 Investor Letter."""
    return INVESTMENT_PHILOSOPHY


@mcp.tool()
def analyze_portfolio_allocation(
    equities_pct: float,
    fixed_income_pct: float,
    alternatives_pct: float,
    cash_pct: float,
) -> str:
    """Analyze a portfolio allocation from a Stone Ridge perspective.

    The analysis uses principles from the 2025 Investor Letter — Bayesian
    diversification, independent risk factors, and tail-risk harvesting.

    Args:
        equities_pct: Percentage allocated to equities (0-100).
        fixed_income_pct: Percentage allocated to fixed income (0-100).
        alternatives_pct: Percentage allocated to alternatives (0-100).
        cash_pct: Percentage allocated to cash (0-100).
    """
    total = equities_pct + fixed_income_pct + alternatives_pct + cash_pct
    if abs(total - 100) > 0.5:
        return f"Allocations sum to {total:.1f}% — they should sum to 100%."

    lines = [
        "## Portfolio Allocation Analysis (Stone Ridge Perspective)\n",
        f"| Asset Class    | Allocation |",
        f"|----------------|------------|",
        f"| Equities       | {equities_pct:.1f}%      |",
        f"| Fixed Income   | {fixed_income_pct:.1f}%      |",
        f"| Alternatives   | {alternatives_pct:.1f}%      |",
        f"| Cash           | {cash_pct:.1f}%      |",
        "",
        "### Stone Ridge Commentary",
        "",
    ]

    if alternatives_pct < 10:
        lines.append(
            "- **Low alternatives allocation**: Stone Ridge advocates for meaningful "
            "exposure to genuinely uncorrelated return streams (reinsurance, energy, "
            "bitcoin).  A sub-10% alternatives allocation may miss diversification "
            "benefits from independent risk factors."
        )
    elif alternatives_pct >= 10 and alternatives_pct < 25:
        lines.append(
            "- **Moderate alternatives allocation**: Reasonable starting point.  "
            "Consider whether your alternatives are truly independent of equity-market "
            "beta — many hedge-fund strategies retain hidden equity correlation."
        )
    else:
        lines.append(
            "- **Substantial alternatives allocation**: Aligns with Stone Ridge's "
            "philosophy of harvesting multiple independent risk premiums.  Ensure "
            "sufficient liquidity to weather tail events."
        )

    if equities_pct > 60:
        lines.append(
            "- **Equity-heavy portfolio**: Equity risk dominates — even a 60/40 "
            "portfolio derives ~90% of its risk from equities.  Stone Ridge "
            "recommends diversifying *risk* not just *capital*."
        )

    if cash_pct > 20:
        lines.append(
            "- **High cash position**: While dry powder is valuable, idle cash "
            "carries opportunity cost.  Stone Ridge's Bayesian framework suggests "
            "deploying into positively-skewed, independent risk premiums."
        )

    return "\n".join(lines)


@mcp.tool()
def search_investor_letter(query: str, section: str = "all") -> str:
    """Keyword search across sections of the Stone Ridge 2025 Investor Letter.

    Args:
        query: Keywords to search for (case-insensitive).
        section: Section to search — 'overview', 'reinsurance', 'bitcoin',
                 'energy', 'philosophy', or 'all' (default).
    """
    query_lower = query.lower()
    results = []

    if section == "all":
        sections_to_search = LETTER_SECTIONS.items()
    else:
        key = section.strip().lower()
        if key in LETTER_SECTIONS:
            sections_to_search = [(key, LETTER_SECTIONS[key])]
        else:
            return (
                f"Section '{section}' not found. "
                f"Available sections: {', '.join(LETTER_SECTIONS.keys())}"
            )

    for sec_name, sec_text in sections_to_search:
        if query_lower in sec_text.lower():
            results.append(f"**[{sec_name.title()}]**\n{sec_text}")

    if not results:
        return f"No results for '{query}' in the investor letter."

    return f"Found matches in {len(results)} section(s):\n\n" + "\n\n---\n\n".join(results)


@mcp.tool()
def compare_funds(fund_a: str, fund_b: str) -> str:
    """Compare two Stone Ridge funds side by side.

    Returns an overview and key themes comparison for the two funds.

    Args:
        fund_a: Name or keyword for the first fund (e.g., 'SRE', 'Bitcoin').
        fund_b: Name or keyword for the second fund (e.g., 'Reinsurance', 'Longtail Re').
    """

    def _resolve(name: str):
        key = name.strip().lower()
        fund = FUNDS.get(key)
        if fund is None:
            for k, v in FUNDS.items():
                if key in k or key in v["name"].lower():
                    return v
        return fund

    a = _resolve(fund_a)
    b = _resolve(fund_b)

    if a is None:
        return f"Fund '{fund_a}' not found. Available: {', '.join(f['name'] for f in FUNDS.values())}"
    if b is None:
        return f"Fund '{fund_b}' not found. Available: {', '.join(f['name'] for f in FUNDS.values())}"

    themes_a = "\n".join(f"  - {t}" for t in a["key_themes"])
    themes_b = "\n".join(f"  - {t}" for t in b["key_themes"])

    return (
        f"## Fund Comparison\n\n"
        f"### {a['name']}\n\n{a['overview']}\n\n**Key Themes:**\n{themes_a}\n\n"
        f"---\n\n"
        f"### {b['name']}\n\n{b['overview']}\n\n**Key Themes:**\n{themes_b}\n\n"
        f"---\n\n"
        f"### Key Differences\n\n"
        f"- **{a['name']}** focuses on: {a['key_themes'][0].lower()}\n"
        f"- **{b['name']}** focuses on: {b['key_themes'][0].lower()}\n"
        f"- Both reflect Stone Ridge's Bayesian investment philosophy and emphasis on "
        f"harvesting independent risk premiums."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
