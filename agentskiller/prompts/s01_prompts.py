"""
Prompts for Step 01: Domain Expansion

Expand seed domains into diverse sub-domains.
"""

# Seed domains to bootstrap the generation process
SEED_DOMAINS = [
    "healthcare",
    "finance",
    "education",
    "e_commerce",
    "travel",
    "real_estate",
    "legal",
    "manufacturing",
    "agriculture",
    "entertainment",
    "sports",
    "government",
    "non_profit",
    "research",
    "media",
]

DOMAIN_EXPANSION_PROMPT = """## Domain Expansion Task

### Core Concept

A **domain** is a self-contained system centered around a specific topic, comprising a set of public or private functions, and providing services under predefined constraints.

- **Good Example**: "airline" — includes functions like book_ticket, cancel_reservation, check_flight_status, etc.
- **Bad Example**: "booking_airline" — too narrow; won't contain diverse functions.
- **Bad Example**: "travel_and_finance" — too broad; use "A_and_B" naming sparingly.

### Your Task

Given the existing domain(s):
```
{existing_domains}
```

**Propose {num_new_domains} NEW domains** that:
1. Are related to the existing domains within a broader real-world context
2. Can be instantiated as a system/application with which users interact to achieve goals
3. Cover diverse aspects of human life and business operations

### Expansion Strategies (Use ALL of them)

1. **Same-Context Expansion** (Horizontal):
   - Given ['hotel', 'weather'] in "Travel" context → propose 'airline', 'car_rental', 'travel_insurance'
   - Think: "What other services would users need in this context?"

2. **Cross-Context Bridging** (Diagonal):
   - Given ['course_management'] in "Education" → bridge to "Travel" via 'student_travel_booking'
   - Given ['healthcare'] → bridge to "Finance" via 'medical_billing', 'health_insurance'
   - Think: "How do users from one domain naturally need services from another?"

3. **Lifecycle Expansion** (Vertical):
   - Given ['job_search'] → propose 'employee_onboarding', 'performance_review', 'retirement_planning'
   - Think: "What comes before/after this domain in a user's lifecycle?"

4. **Specialization** (Depth):
   - Given ['healthcare'] → propose 'dental_services', 'mental_health', 'pediatrics', 'pharmacy'
   - Think: "What are specific sub-types or specialized versions?"

5. **Role-Based Expansion**:
   - Given ['hospital'] → propose 'nurse_scheduling', 'doctor_consultations', 'patient_records'
   - Think: "What different roles interact with this domain?"

### Output Format (STRICT JSON)

```json
{{
    "domains": ["domain1", "domain2", ...],
    "reasoning": "Brief explanation of expansion strategy used"
}}
```

### Constraints
- Use snake_case for domain names
- Each domain name should be 1-3 words
- Avoid overly generic names (e.g., "management", "system")
- Avoid overly specific names (e.g., "monday_morning_coffee_ordering")
"""

DOMAIN_DIVERSITY_CHECK_PROMPT = """## Domain Diversity Check

Review these domains for diversity and coverage:
```
{domains}
```

Identify:
1. Any clusters that are over-represented
2. Missing areas that should be covered
3. Domains that are too similar and could be merged

Suggest {num_suggestions} new domains to improve diversity.

Output as JSON:
```json
{{
    "over_represented": ["cluster1", "cluster2"],
    "under_represented": ["area1", "area2"],
    "similar_pairs": [["domain1", "domain2"]],
    "new_domains": ["new1", "new2", ...]
}}
```
"""
