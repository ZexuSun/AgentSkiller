"""
Abbreviation mappings for equivalence checking.

Uses pre-computed index table for O(1) lookup performance.
This module provides bidirectional mapping between abbreviations and their full forms.

Examples:
    - "FR" ↔ "France"
    - "CA" ↔ "California"
    - "Jan" ↔ "January"
    - "USD" ↔ "US Dollar"
"""
from typing import Dict, List


# =============================================================================
# EQUIVALENCE GROUPS DEFINITION
# Each list contains values that should be considered equivalent.
# All comparisons are case-insensitive.
# =============================================================================

_EQUIVALENCE_GROUPS: List[List[str]] = [
    # =========================================================================
    # COUNTRIES (ISO 3166-1 alpha-2, alpha-3, and common names)
    # =========================================================================
    ["af", "afg", "afghanistan"],
    ["al", "alb", "albania"],
    ["dz", "dza", "algeria"],
    ["ad", "and", "andorra"],
    ["ao", "ago", "angola"],
    ["ag", "atg", "antigua and barbuda"],
    ["ar", "arg", "argentina"],
    ["am", "arm", "armenia"],
    ["au", "aus", "australia"],
    ["at", "aut", "austria"],
    ["az", "aze", "azerbaijan"],
    ["bs", "bhs", "bahamas"],
    ["bh", "bhr", "bahrain"],
    ["bd", "bgd", "bangladesh"],
    ["bb", "brb", "barbados"],
    ["by", "blr", "belarus"],
    ["be", "bel", "belgium"],
    ["bz", "blz", "belize"],
    ["bj", "ben", "benin"],
    ["bt", "btn", "bhutan"],
    ["bo", "bol", "bolivia"],
    ["ba", "bih", "bosnia and herzegovina", "bosnia"],
    ["bw", "bwa", "botswana"],
    ["br", "bra", "brazil"],
    ["bn", "brn", "brunei"],
    ["bg", "bgr", "bulgaria"],
    ["bf", "bfa", "burkina faso"],
    ["bi", "bdi", "burundi"],
    ["cv", "cpv", "cabo verde", "cape verde"],
    ["kh", "khm", "cambodia"],
    ["cm", "cmr", "cameroon"],
    ["ca", "can", "canada"],
    ["cf", "caf", "central african republic"],
    ["td", "tcd", "chad"],
    ["cl", "chl", "chile"],
    ["cn", "chn", "china", "people's republic of china", "prc"],
    ["co", "col", "colombia"],
    ["km", "com", "comoros"],
    ["cg", "cog", "congo"],
    ["cd", "cod", "democratic republic of the congo", "dr congo", "drc"],
    ["cr", "cri", "costa rica"],
    ["ci", "civ", "cote d'ivoire", "ivory coast"],
    ["hr", "hrv", "croatia"],
    ["cu", "cub", "cuba"],
    ["cy", "cyp", "cyprus"],
    ["cz", "cze", "czech republic", "czechia"],
    ["dk", "dnk", "denmark"],
    ["dj", "dji", "djibouti"],
    ["dm", "dma", "dominica"],
    ["do", "dom", "dominican republic"],
    ["ec", "ecu", "ecuador"],
    ["eg", "egy", "egypt"],
    ["sv", "slv", "el salvador"],
    ["gq", "gnq", "equatorial guinea"],
    ["er", "eri", "eritrea"],
    ["ee", "est", "estonia"],
    ["sz", "swz", "eswatini", "swaziland"],
    ["et", "eth", "ethiopia"],
    ["fj", "fji", "fiji"],
    ["fi", "fin", "finland"],
    ["fr", "fra", "france"],
    ["ga", "gab", "gabon"],
    ["gm", "gmb", "gambia"],
    ["ge", "geo", "georgia"],
    ["de", "deu", "germany"],
    ["gh", "gha", "ghana"],
    ["gr", "grc", "greece"],
    ["gd", "grd", "grenada"],
    ["gt", "gtm", "guatemala"],
    ["gn", "gin", "guinea"],
    ["gw", "gnb", "guinea-bissau"],
    ["gy", "guy", "guyana"],
    ["ht", "hti", "haiti"],
    ["hn", "hnd", "honduras"],
    ["hu", "hun", "hungary"],
    ["is", "isl", "iceland"],
    ["in", "ind", "india"],
    ["id", "idn", "indonesia"],
    ["ir", "irn", "iran"],
    ["iq", "irq", "iraq"],
    ["ie", "irl", "ireland"],
    ["il", "isr", "israel"],
    ["it", "ita", "italy"],
    ["jm", "jam", "jamaica"],
    ["jp", "jpn", "japan"],
    ["jo", "jor", "jordan"],
    ["kz", "kaz", "kazakhstan"],
    ["ke", "ken", "kenya"],
    ["ki", "kir", "kiribati"],
    ["kp", "prk", "north korea", "dprk"],
    ["kr", "kor", "south korea", "korea"],
    ["kw", "kwt", "kuwait"],
    ["kg", "kgz", "kyrgyzstan"],
    ["la", "lao", "laos"],
    ["lv", "lva", "latvia"],
    ["lb", "lbn", "lebanon"],
    ["ls", "lso", "lesotho"],
    ["lr", "lbr", "liberia"],
    ["ly", "lby", "libya"],
    ["li", "lie", "liechtenstein"],
    ["lt", "ltu", "lithuania"],
    ["lu", "lux", "luxembourg"],
    ["mg", "mdg", "madagascar"],
    ["mw", "mwi", "malawi"],
    ["my", "mys", "malaysia"],
    ["mv", "mdv", "maldives"],
    ["ml", "mli", "mali"],
    ["mt", "mlt", "malta"],
    ["mh", "mhl", "marshall islands"],
    ["mr", "mrt", "mauritania"],
    ["mu", "mus", "mauritius"],
    ["mx", "mex", "mexico"],
    ["fm", "fsm", "micronesia"],
    ["md", "mda", "moldova"],
    ["mc", "mco", "monaco"],
    ["mn", "mng", "mongolia"],
    ["me", "mne", "montenegro"],
    ["ma", "mar", "morocco"],
    ["mz", "moz", "mozambique"],
    ["mm", "mmr", "myanmar", "burma"],
    ["na", "nam", "namibia"],
    ["nr", "nru", "nauru"],
    ["np", "npl", "nepal"],
    ["nl", "nld", "netherlands", "holland"],
    ["nz", "nzl", "new zealand"],
    ["ni", "nic", "nicaragua"],
    ["ne", "ner", "niger"],
    ["ng", "nga", "nigeria"],
    ["mk", "mkd", "north macedonia", "macedonia"],
    ["no", "nor", "norway"],
    ["om", "omn", "oman"],
    ["pk", "pak", "pakistan"],
    ["pw", "plw", "palau"],
    ["ps", "pse", "palestine"],
    ["pa", "pan", "panama"],
    ["pg", "png", "papua new guinea"],
    ["py", "pry", "paraguay"],
    ["pe", "per", "peru"],
    ["ph", "phl", "philippines"],
    ["pl", "pol", "poland"],
    ["pt", "prt", "portugal"],
    ["qa", "qat", "qatar"],
    ["ro", "rou", "romania"],
    ["ru", "rus", "russia", "russian federation"],
    ["rw", "rwa", "rwanda"],
    ["kn", "kna", "saint kitts and nevis"],
    ["lc", "lca", "saint lucia"],
    ["vc", "vct", "saint vincent and the grenadines"],
    ["ws", "wsm", "samoa"],
    ["sm", "smr", "san marino"],
    ["st", "stp", "sao tome and principe"],
    ["sa", "sau", "saudi arabia"],
    ["sn", "sen", "senegal"],
    ["rs", "srb", "serbia"],
    ["sc", "syc", "seychelles"],
    ["sl", "sle", "sierra leone"],
    ["sg", "sgp", "singapore"],
    ["sk", "svk", "slovakia"],
    ["si", "svn", "slovenia"],
    ["sb", "slb", "solomon islands"],
    ["so", "som", "somalia"],
    ["za", "zaf", "south africa"],
    ["ss", "ssd", "south sudan"],
    ["es", "esp", "spain"],
    ["lk", "lka", "sri lanka"],
    ["sd", "sdn", "sudan"],
    ["sr", "sur", "suriname"],
    ["se", "swe", "sweden"],
    ["ch", "che", "switzerland"],
    ["sy", "syr", "syria"],
    ["tw", "twn", "taiwan"],
    ["tj", "tjk", "tajikistan"],
    ["tz", "tza", "tanzania"],
    ["th", "tha", "thailand"],
    ["tl", "tls", "timor-leste", "east timor"],
    ["tg", "tgo", "togo"],
    ["to", "ton", "tonga"],
    ["tt", "tto", "trinidad and tobago"],
    ["tn", "tun", "tunisia"],
    ["tr", "tur", "turkey", "turkiye"],
    ["tm", "tkm", "turkmenistan"],
    ["tv", "tuv", "tuvalu"],
    ["ug", "uga", "uganda"],
    ["ua", "ukr", "ukraine"],
    ["ae", "are", "united arab emirates", "uae"],
    ["gb", "gbr", "uk", "united kingdom", "great britain", "britain", "england"],
    ["us", "usa", "united states", "united states of america", "america"],
    ["uy", "ury", "uruguay"],
    ["uz", "uzb", "uzbekistan"],
    ["vu", "vut", "vanuatu"],
    ["va", "vat", "vatican city", "holy see"],
    ["ve", "ven", "venezuela"],
    ["vn", "vnm", "vietnam", "viet nam"],
    ["ye", "yem", "yemen"],
    ["zm", "zmb", "zambia"],
    ["zw", "zwe", "zimbabwe"],

    # =========================================================================
    # US STATES AND TERRITORIES
    # =========================================================================
    ["al", "alabama"],
    ["ak", "alaska"],
    ["az", "arizona"],
    ["ar", "arkansas"],
    ["ca", "california", "calif"],
    ["co", "colorado", "colo"],
    ["ct", "connecticut", "conn"],
    ["de", "delaware", "del"],
    ["fl", "florida", "fla"],
    ["ga", "georgia"],
    ["hi", "hawaii"],
    ["id", "idaho"],
    ["il", "illinois", "ill"],
    ["in", "indiana", "ind"],
    ["ia", "iowa"],
    ["ks", "kansas", "kans"],
    ["ky", "kentucky"],
    ["la", "louisiana"],
    ["me", "maine"],
    ["md", "maryland"],
    ["ma", "massachusetts", "mass"],
    ["mi", "michigan", "mich"],
    ["mn", "minnesota", "minn"],
    ["ms", "mississippi", "miss"],
    ["mo", "missouri"],
    ["mt", "montana", "mont"],
    ["ne", "nebraska", "nebr"],
    ["nv", "nevada", "nev"],
    ["nh", "new hampshire"],
    ["nj", "new jersey"],
    ["nm", "new mexico"],
    ["ny", "new york"],
    ["nc", "north carolina"],
    ["nd", "north dakota"],
    ["oh", "ohio"],
    ["ok", "oklahoma", "okla"],
    ["or", "oregon", "ore"],
    ["pa", "pennsylvania", "penn"],
    ["ri", "rhode island"],
    ["sc", "south carolina"],
    ["sd", "south dakota"],
    ["tn", "tennessee", "tenn"],
    ["tx", "texas", "tex"],
    ["ut", "utah"],
    ["vt", "vermont"],
    ["va", "virginia"],
    ["wa", "washington", "wash"],
    ["wv", "west virginia"],
    ["wi", "wisconsin", "wis", "wisc"],
    ["wy", "wyoming", "wyo"],
    ["dc", "district of columbia", "washington dc", "washington d.c."],
    ["pr", "puerto rico"],
    ["vi", "virgin islands", "us virgin islands"],
    ["gu", "guam"],
    ["as", "american samoa"],
    ["mp", "northern mariana islands"],

    # =========================================================================
    # CANADIAN PROVINCES AND TERRITORIES
    # =========================================================================
    ["ab", "alberta", "alta"],
    ["bc", "british columbia"],
    ["mb", "manitoba", "man"],
    ["nb", "new brunswick"],
    ["nl", "newfoundland and labrador", "newfoundland", "nfld"],
    ["nt", "northwest territories", "nwt"],
    ["ns", "nova scotia"],
    ["nu", "nunavut"],
    ["on", "ontario", "ont"],
    ["pe", "prince edward island", "pei"],
    ["qc", "quebec", "que"],
    ["sk", "saskatchewan", "sask"],
    ["yt", "yukon", "yukon territory"],

    # =========================================================================
    # MONTHS
    # =========================================================================
    ["jan", "january"],
    ["feb", "february"],
    ["mar", "march"],
    ["apr", "april"],
    ["may", "may"],
    ["jun", "june"],
    ["jul", "july"],
    ["aug", "august"],
    ["sep", "sept", "september"],
    ["oct", "october"],
    ["nov", "november"],
    ["dec", "december"],

    # =========================================================================
    # DAYS OF THE WEEK
    # =========================================================================
    ["mon", "monday"],
    ["tue", "tues", "tuesday"],
    ["wed", "wednesday"],
    ["thu", "thur", "thurs", "thursday"],
    ["fri", "friday"],
    ["sat", "saturday"],
    ["sun", "sunday"],

    # =========================================================================
    # TITLES AND HONORIFICS
    # =========================================================================
    ["mr", "mr.", "mister"],
    ["mrs", "mrs.", "missus"],
    ["ms", "ms.", "miss"],
    ["dr", "dr.", "doctor"],
    ["prof", "prof.", "professor"],
    ["rev", "rev.", "reverend"],
    ["hon", "hon.", "honorable"],
    ["sr", "sr.", "senior"],
    ["jr", "jr.", "junior"],
    ["esq", "esq.", "esquire"],
    ["capt", "capt.", "captain"],
    ["col", "col.", "colonel"],
    ["gen", "gen.", "general"],
    ["lt", "lt.", "lieutenant"],
    ["sgt", "sgt.", "sergeant"],
    ["cpl", "cpl.", "corporal"],
    ["pvt", "pvt.", "private"],
    ["adm", "adm.", "admiral"],

    # =========================================================================
    # CURRENCY CODES (ISO 4217)
    # =========================================================================
    ["usd", "us dollar", "us dollars", "dollar", "dollars", "$"],
    ["eur", "euro", "euros", "€"],
    ["gbp", "british pound", "pound sterling", "pounds", "£"],
    ["jpy", "japanese yen", "yen", "¥"],
    ["cny", "rmb", "chinese yuan", "yuan", "renminbi"],
    ["cad", "canadian dollar", "canadian dollars"],
    ["aud", "australian dollar", "australian dollars"],
    ["chf", "swiss franc", "swiss francs"],
    ["hkd", "hong kong dollar", "hong kong dollars"],
    ["sgd", "singapore dollar", "singapore dollars"],
    ["sek", "swedish krona", "swedish kronor"],
    ["nok", "norwegian krone", "norwegian kroner"],
    ["dkk", "danish krone", "danish kroner"],
    ["nzd", "new zealand dollar", "new zealand dollars"],
    ["krw", "south korean won", "korean won", "won"],
    ["inr", "indian rupee", "indian rupees", "rupee", "rupees"],
    ["mxn", "mexican peso", "mexican pesos"],
    ["brl", "brazilian real", "brazilian reais", "real", "reais"],
    ["rub", "russian ruble", "russian rubles", "ruble", "rubles"],
    ["zar", "south african rand", "rand"],
    ["thb", "thai baht", "baht"],
    ["idr", "indonesian rupiah", "rupiah"],
    ["myr", "malaysian ringgit", "ringgit"],
    ["php", "philippine peso", "philippine pesos"],
    ["pln", "polish zloty", "zloty"],
    ["try", "turkish lira", "lira"],
    ["aed", "uae dirham", "dirham"],
    ["sar", "saudi riyal", "riyal"],

    # =========================================================================
    # MEASUREMENT UNITS - LENGTH
    # =========================================================================
    ["km", "kilometer", "kilometers", "kilometre", "kilometres"],
    ["m", "meter", "meters", "metre", "metres"],
    ["cm", "centimeter", "centimeters", "centimetre", "centimetres"],
    ["mm", "millimeter", "millimeters", "millimetre", "millimetres"],
    ["mi", "mile", "miles"],
    ["yd", "yard", "yards"],
    ["ft", "foot", "feet"],
    ["in", "inch", "inches"],
    ["nm", "nautical mile", "nautical miles"],

    # =========================================================================
    # MEASUREMENT UNITS - WEIGHT/MASS
    # =========================================================================
    ["kg", "kilogram", "kilograms", "kilo", "kilos"],
    ["g", "gram", "grams"],
    ["mg", "milligram", "milligrams"],
    ["lb", "lbs", "pound", "pounds"],
    ["oz", "ounce", "ounces"],
    ["t", "ton", "tons", "tonne", "tonnes", "metric ton", "metric tons"],
    ["st", "stone", "stones"],

    # =========================================================================
    # MEASUREMENT UNITS - VOLUME
    # =========================================================================
    ["l", "liter", "liters", "litre", "litres"],
    ["ml", "milliliter", "milliliters", "millilitre", "millilitres"],
    ["gal", "gallon", "gallons"],
    ["qt", "quart", "quarts"],
    ["pt", "pint", "pints"],
    ["fl oz", "fluid ounce", "fluid ounces"],
    ["cup", "cups"],
    ["tbsp", "tablespoon", "tablespoons"],
    ["tsp", "teaspoon", "teaspoons"],

    # =========================================================================
    # MEASUREMENT UNITS - TEMPERATURE
    # =========================================================================
    ["c", "°c", "celsius", "centigrade"],
    ["f", "°f", "fahrenheit"],
    ["k", "kelvin"],

    # =========================================================================
    # MEASUREMENT UNITS - SPEED
    # =========================================================================
    ["mph", "miles per hour"],
    ["kph", "km/h", "kmh", "kilometers per hour", "kilometres per hour"],
    ["mps", "m/s", "meters per second", "metres per second"],
    ["fps", "ft/s", "feet per second"],
    ["kn", "kt", "knot", "knots"],

    # =========================================================================
    # MEASUREMENT UNITS - AREA
    # =========================================================================
    ["sq km", "km2", "km²", "square kilometer", "square kilometers", "square kilometre", "square kilometres"],
    ["sq m", "m2", "m²", "square meter", "square meters", "square metre", "square metres"],
    ["sq mi", "mi2", "mi²", "square mile", "square miles"],
    ["sq ft", "ft2", "ft²", "square foot", "square feet"],
    ["ha", "hectare", "hectares"],
    ["ac", "acre", "acres"],

    # =========================================================================
    # TIME UNITS
    # =========================================================================
    ["sec", "s", "second", "seconds"],
    ["min", "minute", "minutes"],
    ["hr", "h", "hour", "hours"],
    ["wk", "week", "weeks"],
    ["mo", "month", "months"],
    ["yr", "y", "year", "years"],

    # =========================================================================
    # COMMON ABBREVIATIONS
    # =========================================================================
    ["no", "no.", "num", "number"],
    ["vol", "vol.", "volume"],
    ["pg", "pg.", "page"],
    ["pp", "pp.", "pages"],
    ["ch", "ch.", "chapter"],
    ["sec", "sec.", "section"],
    ["fig", "fig.", "figure"],
    ["approx", "approx.", "approximately", "about", "circa"],
    ["etc", "etc.", "et cetera"],
    ["ie", "i.e.", "that is"],
    ["eg", "e.g.", "for example"],
    ["vs", "vs.", "versus", "v.", "v"],
    ["dept", "dept.", "department"],
    ["corp", "corp.", "corporation"],
    ["inc", "inc.", "incorporated"],
    ["ltd", "ltd.", "limited"],
    ["co", "co.", "company"],
    ["assn", "assn.", "association"],
    ["intl", "int'l", "international"],
    ["natl", "nat'l", "national"],
    ["govt", "gov't", "government"],
    ["univ", "university"],
    ["st", "st.", "street"],
    ["ave", "ave.", "avenue"],
    ["blvd", "blvd.", "boulevard"],
    ["rd", "rd.", "road"],
    ["dr", "dr.", "drive"],
    ["ln", "ln.", "lane"],
    ["ct", "ct.", "court"],
    ["pl", "pl.", "place"],
    ["sq", "sq.", "square"],
    ["apt", "apt.", "apartment"],
    ["ste", "ste.", "suite"],
    ["fl", "fl.", "floor"],
    ["bldg", "bldg.", "building"],
    ["rm", "rm.", "room"],
    ["po box", "p.o. box", "post office box"],

    # =========================================================================
    # COMPASS DIRECTIONS
    # =========================================================================
    ["n", "north"],
    ["s", "south"],
    ["e", "east"],
    ["w", "west"],
    ["ne", "northeast", "north-east"],
    ["nw", "northwest", "north-west"],
    ["se", "southeast", "south-east"],
    ["sw", "southwest", "south-west"],
    ["nnw", "north-northwest"],
    ["nne", "north-northeast"],
    ["ssw", "south-southwest"],
    ["sse", "south-southeast"],
    ["wnw", "west-northwest"],
    ["wsw", "west-southwest"],
    ["ene", "east-northeast"],
    ["ese", "east-southeast"],

    # =========================================================================
    # ACADEMIC DEGREES
    # =========================================================================
    ["ba", "b.a.", "bachelor of arts"],
    ["bs", "b.s.", "bsc", "b.sc.", "bachelor of science"],
    ["ma", "m.a.", "master of arts"],
    ["ms", "m.s.", "msc", "m.sc.", "master of science"],
    ["mba", "m.b.a.", "master of business administration"],
    ["phd", "ph.d.", "doctor of philosophy"],
    ["md", "m.d.", "doctor of medicine"],
    ["jd", "j.d.", "juris doctor"],
    ["llb", "ll.b.", "bachelor of laws"],
    ["llm", "ll.m.", "master of laws"],
    ["edd", "ed.d.", "doctor of education"],
    ["dds", "d.d.s.", "doctor of dental surgery"],
    ["dvm", "d.v.m.", "doctor of veterinary medicine"],
    ["rn", "r.n.", "registered nurse"],
    ["cpa", "c.p.a.", "certified public accountant"],

    # =========================================================================
    # BOOLEAN / YES-NO
    # =========================================================================
    ["y", "yes", "true", "1"],
    ["n", "no", "false", "0"],

    # =========================================================================
    # GENDER
    # =========================================================================
    ["m", "male", "man"],
    ["f", "female", "woman"],

    # =========================================================================
    # TRANSPORTATION
    # =========================================================================
    ["intl", "international"],
    ["dom", "domestic"],
    ["arr", "arrival", "arriving"],
    ["dep", "departure", "departing"],
    ["flt", "flight"],
    ["pax", "passenger", "passengers"],
]


# =============================================================================
# PRE-COMPUTED INDEX TABLE
# Maps each value (lowercase) to its group ID for O(1) lookup
# =============================================================================

_CANONICAL_INDEX: Dict[str, int] = {}


def _build_index() -> None:
    """
    Build the canonical index at module load time.
    
    Time complexity: O(M) where M is total number of values across all groups
    Space complexity: O(M)
    """
    for group_id, group in enumerate(_EQUIVALENCE_GROUPS):
        for value in group:
            normalized = value.lower().strip()
            # Note: If there are conflicts (same value in multiple groups),
            # the later group wins. This is intentional for some overlapping
            # abbreviations (e.g., "in" could be Indiana or inch).
            _CANONICAL_INDEX[normalized] = group_id


# Build index when module is loaded
_build_index()


def are_equivalent(val1: str, val2: str) -> bool:
    """
    Check if two values are equivalent via abbreviation mapping.
    
    This function performs O(1) lookup using pre-computed index table.
    Both values must be in the mapping table and belong to the same
    equivalence group for this to return True.
    
    Args:
        val1: First value to compare
        val2: Second value to compare
    
    Returns:
        True if both values belong to the same equivalence group,
        False otherwise (including if either value is not in the mapping)
    
    Examples:
        >>> are_equivalent("FR", "France")
        True
        >>> are_equivalent("fr", "FRANCE")
        True
        >>> are_equivalent("CA", "California")
        True
        >>> are_equivalent("Jan", "January")
        True
        >>> are_equivalent("USD", "US Dollar")
        True
        >>> are_equivalent("foo", "bar")
        False
        >>> are_equivalent("FR", "Germany")
        False
    """
    v1_lower = val1.lower().strip()
    v2_lower = val2.lower().strip()
    
    # Fast path: if already equal (case-insensitive), return True
    if v1_lower == v2_lower:
        return True
    
    # Look up both values in the index
    idx1 = _CANONICAL_INDEX.get(v1_lower)
    idx2 = _CANONICAL_INDEX.get(v2_lower)
    
    # Both values must be in the table AND in the same group
    if idx1 is not None and idx2 is not None:
        return idx1 == idx2
    
    return False


def get_equivalents(value: str) -> List[str]:
    """
    Get all equivalent values for a given value.
    
    Args:
        value: The value to look up
    
    Returns:
        List of all equivalent values (including the input if found),
        or empty list if value is not in any equivalence group
    
    Examples:
        >>> get_equivalents("FR")
        ['fr', 'fra', 'france']
        >>> get_equivalents("unknown")
        []
    """
    normalized = value.lower().strip()
    idx = _CANONICAL_INDEX.get(normalized)
    
    if idx is None:
        return []
    
    return _EQUIVALENCE_GROUPS[idx].copy()


def is_known_abbreviation(value: str) -> bool:
    """
    Check if a value is in any equivalence group.
    
    Args:
        value: The value to check
    
    Returns:
        True if the value is in any equivalence group
    
    Examples:
        >>> is_known_abbreviation("FR")
        True
        >>> is_known_abbreviation("xyz123")
        False
    """
    return value.lower().strip() in _CANONICAL_INDEX

