import stock_chemicals as sc
import csv

def vial(chemical, name, number, experiment):
    mass = float(input(f"\tVial {number} {name} mass in grams: "))
    mol_choice = 0.001*float(input(f"\tVial {number} {name} molarity in mM: "))
    mol = mass/chemical
    volume = mol/mol_choice
    volume = volume*1000
    ratio = mol/volume
    #print(name, "with", mass, "g needs", round(volume, 3), "mL water.")
    return mass, mol_choice, volume, ratio #ratio is in mol/L

if __name__ == "__main__":

    plan = input("Enter plan number: ")
    
    with open("Gold (III) Chloride Trihydrate Experiment Plan " + plan + ".csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, dialect='excel', delimiter=',')

        mass_initial = float(input(f"Enter starting mass of AuCl in grams: "))
        mass_used  = 0.0
        experiment = 1

        while mass_used <= mass_initial:
            writer.writerow(["Experiment", experiment, "Remaining Mass [g]:", round(mass_initial, 3)])
            print(f"Experiment {experiment}, remaining mass [g]: {mass_initial}")
            NaCit_YN = input("Is reducing agent NaCit used? (Y/N): ")

            AuCl_mass, AuCl_mol, AuCl_water, AuCl_ratio = vial(sc.HAuCl4_3H2O, "Gold (III) Chloride Trihydrate", 1, experiment)
            writer.writerow(["Vial", "Chemical", "Molarity [mM]", "Mass [g]", "Water added [mL]"])
            writer.writerow([1, "HAuCl4(3H2O)", (AuCl_mol*1000), AuCl_mass, round(AuCl_water, 3)])
            NaCit_mass, NaCit_mol, NaCit_water, NaCit_ratio = 0, 0, 0, 0
            if NaCit_YN == "Y":
                NaCit_mass, NaCit_mol, NaCit_water, NaCit_ratio = vial(sc.NaCit, "Sodium Citrate", 2, experiment)
            writer.writerow([2, "NaCit", (NaCit_mol*1000), NaCit_mass, round(NaCit_water, 3)])

            #Allylamine
            Aam_molarity = 13.33
            Aam_mol = float(input("\tVial 3 Allylamine molarity in mM: "))
            Aam_vial = float(input("\tVial 3 total volume in mL: "))
            irgacure = float(input("\tVial 3a irgacure in g: "))
            Aam_volume = Aam_vial*((Aam_mol*0.001)/Aam_molarity)
            Aam_water = Aam_vial - Aam_volume
            writer.writerow([3, "Allylamine", "1.0 M", str(round(Aam_volume, 3))+" mL", round(Aam_water, 3)])
            writer.writerow(["3a", "PAAM", "-", str(irgacure*1000)+" mg Irg", "-"])
            writer.writerow(["Precursor", "Vial 1 [uL]", "Vial 2 [uL]", "Vial 3a [uL]", "DI Water [uL]", "Total [uL]"])
            vial_1 = float(input("Enter volume of Vial 1 used [uL]: "))
            vial_2 = 0.0
            if NaCit_YN =="Y":
                vial_2 = float(input("Enter volume of Vial 2 used [uL]: "))
            vial_3a = float(input("Enter volume of Vial 3a used [uL]: "))
            DI_water = float(input("Enter volume of DI water added [uL]: "))
            total_precursor = vial_1+vial_2+vial_3a+DI_water
            writer.writerow([" ", vial_1, vial_2, vial_3a, DI_water, total_precursor])
            # M1*L1 = M2*L2 where M is molarity, and L is volume in liters
            gold_mol = 1000*(AuCl_mol*vial_1)/total_precursor
            reducer_mol = 1000*(NaCit_mol*vial_2)/total_precursor
            aam_mol = 1000*vial_3a/total_precursor
            writer.writerow(["HAuCl4(3H2O) [mM]", "NaCit [mM]", "PAAM [mM]"])
            writer.writerow([round(gold_mol, 3), round(reducer_mol, 3), round(aam_mol, 3)])

            mass_initial -= AuCl_mass
            print()
            writer.writerow([" "])
            experiment += 1
            print(f"Remaining AuCl: {round(mass_initial, 3)} g")
