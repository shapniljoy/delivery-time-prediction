# Delivery Time Prediction — EDA & Business Insights Report
**Dataset:** Swiggy food-delivery orders (India) | **N (raw):** 45,593 orders | **N (cleaned):** 45,502 orders | **Target variable:** `time` (delivery time, minutes)

---

## 1. Executive Summary

The data tells a consistent story: **delivery time is driven almost entirely by operational/environmental conditions at the time of the order — traffic, weather, festivals, fleet mix, and batching — not by who the rider is or how far the restaurant is.** That last point is the most important and most counter-intuitive finding in this EDA (see §6.1), and it should be verified and highlighted before any stakeholder presentation, because it reverses the usual assumption that distance is the dominant driver of delivery time.

Top five takeaways for the business:

1. **Traffic is the single strongest lever** — median time roughly doubles from "low" traffic (~20 min) to "jam" (~30 min).
2. **Festivals nearly double delivery time** (25 → 45 min median) but are only 2% of orders — a predictable, schedulable surge, not a chronic problem.
3. **Weather and traffic interact, not just add** — jam traffic under cloudy/foggy skies pushes the median to ~37 minutes, well above what either factor alone would predict.
4. **Distance shows almost no visible relationship with delivery time** — a finding that needs urgent verification (correlation coefficient) because if confirmed, it changes both the operational story and the modeling strategy.
5. **Several categories are dangerously small samples** (bicycles, vehicle_condition = 0, semi-urban city, festival days) — any conclusion drawn about them should be treated as a hypothesis, not a fact, until more data is collected.

---

## 2. Dataset & Methodology Overview

Your `Data_cleaning_utils.perform_data_cleaning()` pipeline transformed the raw 20-column extract into a 25-column analytical table:

- **Rows:** 45,593 → 45,502 (≈91 rows dropped — likely exact duplicates or unsalvageable records)
- **Type coercion:** `Delivery_person_Age` and `Delivery_person_Ratings` arrived as `object` dtype (string-encoded "NaN" sentinels rather than true nulls) and were coerced to numeric — the equivalent of `destring ... , force` in Stata or `pd.to_numeric(..., errors="coerce")` in pandas.
- **Feature engineering:** `distance_km` (Haversine distance from the four lat/long columns), `day`, `month`, `day_of_week`, `is_weekend` (parsed from `Order_Date`), `pickup_time` & `order_time_hour` (parsed from `Time_Orderd`/`Time_Order_picked`), and `time_of_day` (binned from order hour) — plus `city_type`/`city_name` parsed out of the ID fields.


## 3. Data Quality at a Glance

| Field | % Missing | Note |
|---|---|---|
| `age` | 4.1% | coerced from string |
| `ratings` | 4.2% | coerced from string |
| `restaurant_lat/long`, `delivery_lat/long`, `distance_km` | 8.0% | missing together (see above) |
| `weather` | 1.0% | |
| `traffic` | 1.0% | |
| `multiple_deliveries` | 2.2% | |
| `festival` | 0.5% | |
| `city_type` | 2.6% | |
| `pickup_time` / `order_time_hour` | 3.6% | |
| `time_of_day` | 4.5% | |
| `time` (target) | 0.0% | fully observed |

Nothing here is alarming (all under ~8%), but **a methodological flag is worth raising** before you trust the "normality" and "outlier" conclusions printed by your `uni_num()` function:

---

## 4. Univariate Findings

### Numerical variables

| Variable | Mean | Median | Std | Skew | Kurtosis | Read |
|---|---|---|---|---|---|---|
| `age` | 29.6 | 30 | 5.76 | −0.01 | −1.21 | Flat/uniform across 20–39; platykurtic (no peak) |
| `ratings` | 4.64 | 4.7 | 0.31 | −1.79 | 5.14 | Heavily left-skewed; small but real cluster of low-rated riders (≤4.0) |
| `time` (target) | 26.3 | 26 | 9.39 | 0.49 | −0.31 | Bimodal (peaks ~17–18 min and ~26–27 min), long right tail to 54 min |
| `distance_km` | 9.72 | 9.19 | 5.60 | 0.32 | −0.91 | Bimodal with a **gap between 13–16 km** — see §6.1 |

### Categorical variables (share of orders)

| Variable | Distribution |
|---|---|
| `traffic` | low 34%, jam 31%, medium 24%, high 10% |
| `weather` | ~equal across 6 conditions (16–17% each) — looks engineered/balanced rather than naturally observed |
| `vehicle_type` | motorcycle 58%, scooter 34%, e-scooter 8%, **bicycle 0.1%** |
| `vehicle_condition` | conditions 1/2/3 ≈ 33% each, **condition 0 just 0.9%** |
| `multiple_deliveries` | 1 order 63%, 0 orders (solo trip, no batching) 32%, 2 orders 4.5%, 3 orders 0.8% |
| `festival` | no 98%, **yes 2%** |
| `city_type` | metropolitan 77%, urban 23%, **semi-urban 0.4%** |
| `order_type` | snack/meal/drinks/buffet ≈ 25% each — balanced |
| `is_weekend` | weekday 75%, weekend 25% |
| `time_of_day` | evening 42%, night 21%, morning 20%, afternoon 17%, after-midnight ≈0% |
| `day` of month | **day 3 (28%) and day 4 (14%) alone = 42%** of all orders; everything else thinly spread |
| `month` | **Feb + March = ~63%** of all orders |

The `day`/`month` concentration is unusual enough that it deserves its own callout — see §6.4.

---

## 5. What Drives Delivery Time? (Bivariate)

All categorical-vs-time comparisons below were tested with ANOVA (3+ groups) or a t-test (2 groups). Statistical significance (p < 0.05) was found for every variable **except `order_type` (p = 0.225)** and `pickup_time` (p = 0.188) — both can reasonably be excluded as predictors.

**A. Operational / environmental conditions — the biggest levers**
- **Traffic:** median time climbs from ~20 min (low) → ~30 min (jam). This is the strongest single categorical driver in the dataset.
- **Weather:** medians look deceptively similar across conditions, but the ANOVA is significant — the real story is in the *tails*, not the *center* (sunny weather shows extreme high-traffic outliers; stormy/sandstorm/windy behave alike enough to be grouped as "adverse weather").

**B. Fleet & rider factors**
- **Vehicle type:** electric scooters are fastest, motorcycles slowest *and* most variable — but motorcycles are 58% of the fleet and bicycles are only 0.1% of orders, so the bicycle comparison is not reliable.
- **Vehicle condition:** medians are similar across conditions 1–3; condition 0 has too little data (0.9%) to interpret confidently.
- **Rider age / ratings:** no visible bivariate pattern with time — but see §6.2 for why this conclusion deserves a second look before you write it off.

**C. Demand-side factors**
- **Multiple deliveries (batching):** time rises steadily as batch size increases (1 → 2 → 3 stops), with 2–3-stop orders landing in the 35–50 min range. Classic operations trade-off — see §7.
- **Festival:** median time jumps from 25 → 45 min, but it's only 2% of volume — a forecastable spike, not a daily problem.
- **City type:** similar central tendency across metro/urban, but semi-urban is too sparse (0.4%) to trust.

**D. Temporal factors**
- **Day of week:** Thursday has the most orders, Saturday the fewest; delivery time itself is fairly stable across days.
- **Time of day:** evening is both the highest-volume (42% of orders) *and* the slowest (median ~28–29 min) window; morning is both lowest-volume and fastest (~20–21 min). This is a textbook supply/demand mismatch.
- **Weekend:** marginally faster than weekdays (lower order density, likely less traffic).
- **Pickup time:** no effect — restaurant-side pickup speed doesn't move the needle on total delivery time in this data.

---

## 6. Additional Insights

These are patterns I found by going back into the visuals and outputs that weren't called out in your written observations. I'd treat #1 as the highest priority to verify.

### 6.1 — Distance shows almost no visible relationship with delivery time (needs urgent verification)
Your `distance_km` vs. `time` scatterplot is striking: at *every* distance value, delivery time is spread uniformly across the full 10–54 minute range, with no upward slope. For a real-world delivery service this would be highly unusual — distance is normally the dominant driver of delivery time. Two explanations are plausible, and they have very different implications:

- **If genuine:** operational factors (traffic, weather, batching, festival) are doing essentially all the work, and route/distance optimization would yield little ROI compared to managing those operational levers.
- **If a data artifact:** this dataset may be (partly) synthetically generated, with `time` simulated from traffic/weather/festival rather than computed from actual GPS routes — which would mean distance shouldn't be trusted as a real-world feature at all.

**Action:** run `cleaned_df[['distance_km','time']].corr()` (Pearson) and a hexbin/regression-line plot. Visually I'd expect r to come out well under 0.15. Whatever the number is, it's one of the most decision-relevant numbers in this entire analysis and deserves a headline slide of its own.

### 6.2 — Don't conclude "rider age/rating don't matter" from the raw scatterplot alone
A noisy scatterplot is a weak tool for detecting a real but modest relationship, and there's a confounding risk specific to two-sided marketplaces: **if the platform's dispatch algorithm assigns harder orders (longer distance, worse traffic) to higher-rated/more experienced riders**, any true "better riders are faster" effect could be masked or reversed in the raw correlation (an omitted-variable problem, exactly like assigning treatment non-randomly in a quasi-experiment). Before concluding "no effect," I'd bin riders into rating quartiles and compare *conditional* on traffic and distance — or simply add age/ratings to a regression alongside traffic/distance and check whether the coefficient is significant once those confounders are controlled for.

### 6.3 — The low-rated rider segment is small but real, and worth a targeted look
The ratings boxplot shows a distinct cluster of outlier riders rated between 2.5 and ~4.0 — small in count, but not noise. Worth a dedicated cut: do these riders have systematically higher delivery times or more late deliveries? If so, this is a direct lever for rider coaching/quality management.

### 6.4 — The day-of-month and month concentration is probably a data artifact, not seasonality
42% of all orders fall on just two calendar days (3rd and 4th of the month) and 63% fall in just two months (Feb–March). Real demand is never this concentrated. This pattern looks like an artifact of how/when this extract was pulled (e.g., a short collection window or a scrape biased toward certain dates) rather than genuine business seasonality. **Recommendation:** don't present `day`/`month` to stakeholders as "seasonality insights," and be cautious about using raw `day`/`month` as model features — they may not generalize to new data and could quietly hurt model performance in production (a classic train/serve skew risk). `day_of_week`, `is_weekend`, and `time_of_day` are safer, more durable temporal features.

### 6.5 — Weather and traffic *interact* — they don't just add up
Looking at the multivariate weather × traffic chart: jam traffic under **cloudy or foggy** weather pushes the median to ~37 minutes — noticeably higher than jam traffic under sunny/stormy/windy conditions (~21–29 min). That's a super-additive interaction, not two independent effects stacking linearly. **For the predictive model:** include an explicit `traffic × weather` interaction term (or let a tree-based model pick it up natively) rather than relying on main effects alone.

### 6.6 — Large-N hypothesis testing caveat (relevant given your econometrics background)
You ran roughly 14 ANOVA/t-tests across categorical predictors, all at α = 0.05 with no multiple-comparison correction. With n ≈ 45,000, even economically trivial differences will register as "statistically significant" — and 12 of your 14 tests returned p < 0.0001, which is consistent with that large-N effect as much as with large true effects. Practically this doesn't overturn any conclusion here (the two null results — `order_type` and `pickup_time` — stay null under any reasonable correction), but going forward, **pair every p-value with an effect size** (η² for ANOVA, Cohen's d for two groups) so "statistically significant" and "business-relevant" don't get conflated when you write this up for stakeholders.

### 6.7 — Small-sample categories to flag, not hide
Several "significant" or "interesting" findings rest on very thin slices of data: bicycles (0.12% of orders), vehicle_condition = 0 (0.9%), semi-urban city (0.4%), festival days (2%). All are legitimate to report, but each should carry a visible "n is small" caveat — both in this report and later in the Power BI dashboard (see §3 of the Power BI guidance below) — so a viewer doesn't over-index on a 50-order subgroup.

---

## 7. Business Recommendations

- **Dynamic SLA / ETA communication:** Replace a single flat ETA promise with a traffic-and-weather-aware ETA range. The data supports an explicit rule set (e.g., low traffic ≈ 20 min, jam ≈ 30 min, jam + adverse weather ≈ 35–40 min).
- **Festival staffing playbook:** Since festival days are predictable on a calendar and reliably ~80% slower, pre-position extra riders and notify customers proactively rather than reacting after SLA breaches start.
- **Batching policy review:** Multiple-deliveries clearly trades per-order speed for fleet efficiency. Worth an explicit cost/benefit cut: is the cost saved per batched trip worth the delivery-time (and customer satisfaction) hit for orders 2 and 3 in a batch?
- **Fleet mix:** Electric scooters show the best time profile with the least variance — worth investigating as a target for fleet expansion, distance/feasibility permitting.
- **Evening surge:** Evening is simultaneously the highest-demand and slowest window — classic dynamic-dispatch/dynamic-incentive opportunity to pull riders into that window.
- **Data collection fixes:** prioritize closing the GPS coordinate gap (8% of orders) since it silently disables the distance feature, and extend data collection across a full year before treating any month/day pattern as real seasonality.

---

## 8. Implications for the Predictive Model

- **Verify, don't assume, that `distance_km` is a weak predictor** (§6.1) before deciding how much engineering effort to invest in it.
- **Build an explicit `traffic × weather` interaction feature** rather than relying solely on main effects.
- **Avoid using raw `day`/`month` as features** given the likely collection artifact (§6.4); prefer `day_of_week`, `is_weekend`, `time_of_day`.
- **Collapse rare categories** before modeling: merge `multiple_deliveries` 2 & 3 (as you already proposed), consider grouping bicycle into "other," and treat `vehicle_condition = 0` and `city_type = semi-urban` with caution (wide confidence intervals on any coefficient/leaf involving them).
- **Target variable (`time`)** is only mildly right-skewed (skew 0.49) with low kurtosis — a tree-based model (gradient boosting / random forest) should handle it natively without needing a log transform; if you do try a linear/regularized baseline, a mild transform or quantile-based loss is worth testing given the bimodality.
- **Check multicollinearity** between `distance_km`, `city_type`, and `weather` before any linear modeling — the gapped, bimodal shape of `distance_km` (§6.1) hints it may be tightly coupled to `city_type` (e.g., metro vs. non-metro service radii), which would inflate VIFs in a regression context.


