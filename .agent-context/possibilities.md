# Paid / Proprietary Options for Later Consideration

Deferred paid tools and services. Review when budget allows or free alternatives prove insufficient.

## Odds Data APIs

### The Odds API (theoddsapi.com)
- **Cost**: $0–$300/month depending on tier (free tier: 500 requests/month)
- **What it provides**: Real-time odds from Pinnacle, Betfair, DraftKings, and 40+ books via REST API; historical odds snapshots
- **Why deferred**: Free tier too limited for automated daily predictions; paid tiers add recurring cost
- **When to revisit**: If OddsPortal scraping becomes unreliable or we need multi-book line shopping at scale

### Betfair Historical Data
- **Cost**: ~$50–$500/month depending on granularity
- **What it provides**: Tick-by-tick exchange price history, matched volume, opening vs. closing line timestamps
- **Why deferred**: High cost; tennis-data.co.uk provides adequate Pinnacle/B365 historical odds for free
- **When to revisit**: If we want to do matched betting or lay strategies on the exchange at scale

### OddsJam / SBRodds
- **Cost**: $50–$200/month
- **What it provides**: Odds aggregation across all US + international books; closing line tracking; arbitrage alerts
- **Why deferred**: Overkill for initial paper-trading phase
- **When to revisit**: After paper-trading confirms positive ROI and we scale to real money

### Smart Odds / OddsPortal Premium
- **Cost**: ~$30–$100/month
- **What it provides**: Structured historical odds access without scraping ToS issues; reliable uptime
- **Why deferred**: Free scraping is sufficient for development
- **When to revisit**: If scraping breaks or we need guaranteed data SLA

## Compute / Infrastructure

### Cloud GPU (AWS/GCP/Azure)
- **Cost**: $0.30–$3/hour for T4/V100 instances
- **What it provides**: Faster model training; scalable inference
- **Why deferred**: Current dataset (~172K rows, 36 features) trains quickly on CPU/local GPU
- **When to revisit**: If we add recurrent architectures (LSTM/Transformer) or expand to WTA + Challengers significantly

## Data Sources

### Ultimate Tennis Statistics Pro
- **Cost**: Unknown (contact for licensing)
- **What it provides**: Detailed point-by-point statistics, serve speed, rally length
- **Why deferred**: Free tier/public data is sufficient initially
- **When to revisit**: If feature engineering hits a ceiling and we need richer match statistics

## Betting Accounts

### Pinnacle Sports Account
- **Cost**: No subscription; requires depositing real money to use live API
- **What it provides**: Best odds in the world; accepts winning bettors; full API access
- **Why deferred**: Need to validate alpha via paper-trading before real money
- **When to revisit**: After 3+ months of paper-trading shows positive CLV and ROI > 5%

### Betfair Exchange Account
- **Cost**: 2–5% commission on net winnings per market
- **What it provides**: Peer-to-peer odds (often better than Pinnacle); lay betting capability; API access
- **Why deferred**: Commission complicates early-stage ROI calculations; paper-trading first
- **When to revisit**: Once model is validated; potentially lower effective cost than Pinnacle for large volumes
