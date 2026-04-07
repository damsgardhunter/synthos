"""
Curated training data for GNN + XGBoost training pipeline.
Matches the Colab notebook's KNOWN_TC, HYDRIDE_PRESSURE_GPA, and PRESSURE_TC_DATA.
"""

KNOWN_TC = {
    # Cuprates - Lanthanum family
    "La2CuO4": 0.0, "La2BaCuO4": 35.0, "La1.85Sr0.15CuO4": 38.0,
    "La1.85Ba0.15CuO4": 35.0, "La1.8Sr0.2CuO4": 37.0, "Nd2CuO4": 30.0, "Pr2CuO4": 0.0,
    # Cuprates - YBCO family
    "YBa2Cu3O7": 92.0, "YBa2Cu3O6": 0.0, "YBa2Cu4O8": 80.0, "Y2Ba4Cu7O15": 93.0,
    "GdBa2Cu3O7": 94.0, "EuBa2Cu3O7": 95.0, "SmBa2Cu3O7": 92.0, "NdBa2Cu3O7": 90.0,
    "HoBa2Cu3O7": 92.0, "DyBa2Cu3O7": 92.0, "ErBa2Cu3O7": 92.0, "TmBa2Cu3O7": 88.0,
    "LuBa2Cu3O7": 88.0,
    # Cuprates - Bismuth family
    "Bi2Sr2CuO6": 20.0, "Bi2Sr2CaCu2O8": 85.0, "Bi2Sr2Ca2Cu3O10": 110.0, "Bi2Sr2Ca3Cu4O12": 90.0,
    # Cuprates - Thallium family
    "Tl2Ba2CuO6": 90.0, "Tl2Ba2CaCu2O8": 110.0, "Tl2Ba2Ca2Cu3O10": 125.0,
    "TlBa2CuO5": 50.0, "TlBa2CaCu2O7": 80.0, "TlBa2Ca2Cu3O9": 110.0, "TlBa2Ca3Cu4O11": 122.0,
    # Cuprates - Mercury family (record holders)
    "HgBa2CuO4": 94.0, "HgBa2CaCu2O6": 128.0, "HgBa2Ca2Cu3O8": 133.0, "HgBa2Ca3Cu4O10": 125.0,
    # Cuprates - Other
    "RuSr2GdCu2O8": 40.0, "CaCuO2": 110.0, "SrCuO2": 90.0,
    # Iron-based - LaFeAsO family
    "LaFeAsO0.9F0.1": 26.0, "SmFeAsO0.9F0.1": 55.0, "NdFeAsO0.9F0.1": 52.0,
    "CeFeAsO0.9F0.1": 41.0, "GdFeAsO0.9F0.1": 36.0, "LaFePO": 6.0,
    # Iron-based - BaFe2As2 family (122)
    "Ba0.6K0.4Fe2As2": 38.0, "BaFe1.8Co0.2As2": 22.0, "KFe2As2": 3.8, "CsFe2As2": 2.6,
    # Iron-based - FeSe
    "FeSe": 8.0, "FeSe0.5Te0.5": 14.0, "LiFeAs": 18.0, "NaFeAs": 9.0,
    "FeSe_monolayer": 65.0,
    # Pnictides (non-iron)
    "LaRhP": 4.0, "LaRuP": 7.0, "BaNi2As2": 0.7, "SrPtAs": 2.4,
    # Hydrides - Sulfur/Phosphorus
    "H3S": 203.0, "H2S": 80.0, "PH3": 100.0,
    # Hydrides - Lanthanum
    "LaH10": 250.0, "LaH6": 90.0, "LaBeH8": 185.0,
    # Hydrides - Yttrium
    "YH6": 224.0, "YH9": 243.0, "YH4": 84.0,
    # Hydrides - Calcium
    "CaH6": 210.0,
    # Hydrides - Strontium / Barium
    "SrH6": 156.0, "SrH10": 259.0, "BaH6": 38.0, "BaH12": 20.0,
    # Hydrides - Thorium
    "ThH10": 159.0, "ThH9": 146.0,
    # Hydrides - Cerium
    "CeH9": 57.0, "CeH10": 115.0,
    # Hydrides - Ternary
    "LaYH20": 253.0, "LaCeH9": 178.0, "YCeH10": 115.0, "Li2MgH16": 473.0,
    "NbH4": 58.0, "TaH3": 136.0, "AcH16": 251.0,
    # Hydrides - Other
    "MgH6": 263.0, "SiH4": 17.0, "GeH4": 64.0, "SnH4": 70.0, "AlH3": 12.0,
    "LiH6": 82.0, "NaH3": 40.0, "ScH9": 233.0, "ScH6": 169.0,
    "ZrH5": 10.6, "TiH2": 4.15, "PdH": 9.0, "PtH": 12.0,
    # Conventional BCS / A15
    "Nb3Sn": 18.3, "Nb3Ge": 23.2, "Nb3Al": 18.8, "V3Si": 17.1, "V3Ga": 16.8,
    "MgB2": 39.0, "NbN": 16.0, "NbC": 11.1, "MoC": 14.3,
    # Borides
    "YB6": 7.1, "ZrB12": 6.0, "TaB2": 9.5, "OsB2": 2.1,
    # Heavy fermion
    "UBe13": 0.85, "UPt3": 0.54, "CeCoIn5": 2.3, "CeRhIn5": 2.1, "PuCoGa5": 18.5,
    # Other notable
    "Sr2RuO4": 1.5, "K3C60": 19.0, "Rb3C60": 29.0, "Cs3C60": 38.0, "CaC6": 11.5,
    # Binary hydrides (low/zero Tc — important negatives for MP matching)
    "ScH2": 0.0, "ZrH2": 4.0, "YH2": 0.0, "LaH2": 0.0, "CeH3": 0.0, "PrH3": 0.0,
    "NdH3": 0.0, "SmH2": 0.0, "GdH3": 0.0, "DyH3": 0.0, "HoH3": 0.0, "ErH3": 0.0,
    "TmH3": 0.0, "LuH3": 0.0, "VH": 0.0, "NbH": 1.0, "TaH": 0.0, "CrH": 0.0,
    # More cuprate parents / non-SC variants (MP matching)
    "LaCuO3": 0.0, "YCuO3": 0.0, "BiCuO3": 0.0, "CuO2": 0.0,
    "LiCuO2": 0.0, "LiCu2O2": 0.0, "CuCrO2": 0.0, "ScCuO2": 0.0,
    "Gd2CuO4": 0.0, "Eu2CuO4": 20.0, "Sm2CuO4": 0.0,
    # Elemental superconductors
    "Nb": 9.3, "V": 5.4, "Pb": 7.2, "Sn": 3.7, "In": 3.4, "Al": 1.2,
    "Ta": 4.5, "La": 6.0, "Hg": 4.15, "Re": 1.7, "Mo": 0.92,
    "Zr": 0.61, "Ti": 0.4, "W": 0.015, "Ir": 0.11, "Os": 0.66,
    # More A15 and intermetallics
    "Nb3Si": 19.0, "V3Ge": 6.0, "Cr3Si": 0.0, "Mo3Si": 1.3,
    "NbTi": 10.0, "NbZr": 11.0,
    # Chalcogenides
    "NbSe2": 7.2, "NbS2": 6.0, "TaS2": 0.8, "TaSe2": 0.15,
    "MoS2": 0.0, "WS2": 0.0, "PdTe2": 1.7,
    # Nickelate superconductors
    "NdNiO2": 15.0, "LaNiO2": 0.0, "PrNiO2": 12.0,
    # Bismuthides
    "NiBi3": 4.25, "PtBi2": 1.2, "RhBi": 2.06,
}

HYDRIDE_PRESSURE_GPA = {
    "H3S": 155.0, "H2S": 90.0, "PH3": 200.0,
    "LaH10": 170.0, "LaH6": 150.0, "LaBeH8": 120.0,
    "YH6": 166.0, "YH9": 201.0, "YH4": 120.0,
    "CaH6": 170.0, "SrH6": 200.0, "SrH10": 300.0,
    "ThH10": 170.0, "ThH9": 170.0, "CeH9": 150.0, "CeH10": 200.0,
    "LaYH20": 183.0, "LaCeH9": 180.0, "YCeH10": 190.0,
    "Li2MgH16": 250.0, "NbH4": 62.0, "TaH3": 200.0, "AcH16": 200.0,
    "ScH9": 200.0, "ScH6": 130.0, "MgH6": 300.0, "SiH4": 250.0,
    "GeH4": 220.0, "SnH4": 200.0, "LiH6": 150.0, "NaH3": 200.0,
}

# (formula, pressure_gpa, tc_K)
PRESSURE_TC_DATA = [
    # LaH10
    ("LaH10", 0.0, 0.0), ("LaH10", 80.0, 5.0), ("LaH10", 100.0, 50.0),
    ("LaH10", 110.0, 90.0), ("LaH10", 120.0, 130.0), ("LaH10", 130.0, 150.0),
    ("LaH10", 140.0, 175.0), ("LaH10", 150.0, 210.0), ("LaH10", 160.0, 240.0),
    ("LaH10", 170.0, 250.0), ("LaH10", 185.0, 240.0), ("LaH10", 200.0, 220.0),
    # H3S
    ("H3S", 0.0, 0.0), ("H3S", 90.0, 20.0), ("H3S", 100.0, 80.0),
    ("H3S", 110.0, 120.0), ("H3S", 120.0, 150.0), ("H3S", 130.0, 170.0),
    ("H3S", 140.0, 190.0), ("H3S", 150.0, 200.0), ("H3S", 155.0, 203.0),
    ("H3S", 165.0, 195.0), ("H3S", 180.0, 180.0),
    # YH6
    ("YH6", 0.0, 0.0), ("YH6", 80.0, 10.0), ("YH6", 100.0, 80.0),
    ("YH6", 120.0, 140.0), ("YH6", 140.0, 185.0), ("YH6", 150.0, 205.0),
    ("YH6", 160.0, 218.0), ("YH6", 166.0, 224.0), ("YH6", 180.0, 215.0),
    ("YH6", 200.0, 200.0),
    # YH9
    ("YH9", 0.0, 0.0), ("YH9", 100.0, 60.0), ("YH9", 130.0, 130.0),
    ("YH9", 150.0, 185.0), ("YH9", 175.0, 225.0), ("YH9", 201.0, 243.0),
    ("YH9", 220.0, 230.0),
    # CaH6
    ("CaH6", 0.0, 0.0), ("CaH6", 80.0, 10.0), ("CaH6", 100.0, 80.0),
    ("CaH6", 120.0, 130.0), ("CaH6", 140.0, 165.0), ("CaH6", 150.0, 185.0),
    ("CaH6", 160.0, 200.0), ("CaH6", 170.0, 210.0), ("CaH6", 185.0, 200.0),
    # ThH10
    ("ThH10", 0.0, 0.0), ("ThH10", 80.0, 15.0), ("ThH10", 100.0, 60.0),
    ("ThH10", 130.0, 110.0), ("ThH10", 150.0, 140.0), ("ThH10", 170.0, 159.0),
    ("ThH10", 190.0, 145.0),
    # ThH9
    ("ThH9", 0.0, 0.0), ("ThH9", 100.0, 50.0), ("ThH9", 140.0, 110.0),
    ("ThH9", 170.0, 146.0),
    # ScH9
    ("ScH9", 0.0, 0.0), ("ScH9", 100.0, 60.0), ("ScH9", 130.0, 120.0),
    ("ScH9", 150.0, 165.0), ("ScH9", 175.0, 210.0), ("ScH9", 200.0, 233.0),
    # ScH6
    ("ScH6", 0.0, 0.0), ("ScH6", 80.0, 30.0), ("ScH6", 100.0, 90.0),
    ("ScH6", 115.0, 140.0), ("ScH6", 130.0, 169.0),
    # CeH9
    ("CeH9", 0.0, 0.0), ("CeH9", 80.0, 10.0), ("CeH9", 100.0, 30.0),
    ("CeH9", 120.0, 45.0), ("CeH9", 150.0, 57.0), ("CeH9", 170.0, 52.0),
    # CeH10
    ("CeH10", 0.0, 0.0), ("CeH10", 100.0, 40.0), ("CeH10", 150.0, 85.0),
    ("CeH10", 180.0, 110.0), ("CeH10", 200.0, 115.0),
    # LaBeH8
    ("LaBeH8", 0.0, 0.0), ("LaBeH8", 80.0, 20.0), ("LaBeH8", 100.0, 100.0),
    ("LaBeH8", 110.0, 150.0), ("LaBeH8", 120.0, 185.0), ("LaBeH8", 130.0, 175.0),
    # MgB2 - pressure DECREASES Tc
    ("MgB2", 0.0, 39.0), ("MgB2", 1.0, 38.5), ("MgB2", 2.0, 37.8),
    ("MgB2", 5.0, 37.0), ("MgB2", 8.0, 35.5), ("MgB2", 10.0, 34.0),
    ("MgB2", 15.0, 31.0), ("MgB2", 20.0, 28.0), ("MgB2", 25.0, 24.0),
    ("MgB2", 30.0, 20.0),
    # Nb3Sn - modest negative pressure dependence
    ("Nb3Sn", 0.0, 18.3), ("Nb3Sn", 2.0, 17.8), ("Nb3Sn", 5.0, 17.0),
    ("Nb3Sn", 8.0, 16.0), ("Nb3Sn", 10.0, 15.5), ("Nb3Sn", 15.0, 13.5),
    # FeSe - pressure INCREASES Tc dramatically
    ("FeSe", 0.0, 8.0), ("FeSe", 0.5, 13.0), ("FeSe", 1.0, 18.0),
    ("FeSe", 1.5, 23.0), ("FeSe", 2.0, 27.0), ("FeSe", 3.0, 31.0),
    ("FeSe", 4.0, 35.0), ("FeSe", 5.0, 36.5), ("FeSe", 6.0, 37.0),
    ("FeSe", 7.0, 36.5), ("FeSe", 8.5, 36.7), ("FeSe", 10.0, 34.0),
    # YBCO - weak positive pressure dependence
    ("YBa2Cu3O7", 0.0, 92.0), ("YBa2Cu3O7", 2.0, 92.5), ("YBa2Cu3O7", 5.0, 93.0),
    ("YBa2Cu3O7", 10.0, 94.0), ("YBa2Cu3O7", 20.0, 90.0), ("YBa2Cu3O7", 30.0, 86.0),
    # HgBa2Ca2Cu3O8 - large positive pressure effect
    ("HgBa2Ca2Cu3O8", 0.0, 133.0), ("HgBa2Ca2Cu3O8", 10.0, 148.0),
    ("HgBa2Ca2Cu3O8", 20.0, 158.0), ("HgBa2Ca2Cu3O8", 31.0, 164.0),
    ("HgBa2Ca2Cu3O8", 45.0, 150.0),
    # BaFe2As2 - pressure induces superconductivity
    ("BaFe2As2", 0.0, 0.0), ("BaFe2As2", 3.0, 20.0), ("BaFe2As2", 5.5, 29.0),
    ("BaFe2As2", 8.0, 22.0), ("BaFe2As2", 12.0, 5.0),
    # CaFe2As2
    ("CaFe2As2", 0.0, 0.0), ("CaFe2As2", 0.5, 17.0), ("CaFe2As2", 1.0, 12.0),
    # LaFeAsO - pressure enhances Tc
    ("LaFeAsO", 0.0, 0.0), ("LaFeAsO", 3.0, 7.0), ("LaFeAsO", 6.0, 21.0),
    ("LaFeAsO", 13.0, 43.0),
]
