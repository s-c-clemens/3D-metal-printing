# molecular mass in grams per mole (g/mol)
molmass = {"H": 1.00784,
           "C": 12.0107,
           "N": 14.0067,
           "O": 15.9994,
           "Na": 22.9897,
           "Cl": 35.453,
           "Fe": 55.845,
           "Ag": 107.868,
           "Pt": 195.084,
           "Au": 196.966          
}

# metal salts:
AgNO3 = molmass["Ag"] + molmass["N"] + 3*molmass["O"]
AuCl = molmass["Au"] + molmass["Cl"]
HAuCl4_3H2O = 7*molmass["H"] + molmass["Au"] + 3*molmass["O"] + 4*molmass["Cl"]
NH4_2PtCl4 = 2*molmass["N"] + 8*molmass["H"] + molmass["Pt"] + 4*molmass["Cl"]
C6FeN3O12_3H2O = 6*molmass["C"] + 15*molmass["O"] + 18*molmass["H"] + molmass["Fe"] + 3*molmass['N'] #FIXME: wrong MW
# reducing agent:
NaCit = 9*molmass["H"] + 9*molmass["O"] + 6*molmass["C"] + 3*molmass["Na"]
# photoinitiator:
Irgacure = 16*molmass["H"] + 12*molmass["C"] + 4*molmass["O"]

Aam = 7*molmass["H"] + 3*molmass["C"] + molmass["N"]
