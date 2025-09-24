from typing import Literal

CountryShortNameLower = Literal["kw", "ae", "bh", "om", "sa", "lb", "qa", "jo", "eg", "iq"]
CountryShortNameUpper = Literal["KW", "AE", "BH", "OM", "SA", "LB", "QA", "JO", "EG", "IQ"]
CountryShortName = CountryShortNameUpper | CountryShortNameLower
