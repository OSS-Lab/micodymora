import re

class Chem:
    element_pat = re.compile("([A-Z](?:[a-z]*))([0-9]*)")
    charge_pat = re.compile("([+-])([0-9]*)$")

    def __init__(self, name, formula, dGf0):
        self.name = name
        self.dGf0 = dGf0
        self.composition = {element: amount and int(amount) or 1 for element, amount in Chem.element_pat.findall(formula)}
        charge_string = Chem.charge_pat.findall(formula)
        if charge_string:
            sign, charge = charge_string[0]
            self.composition["charge"] = int(sign + (charge or "1"))
        else:
            self.composition["charge"] = 0

    def formula(self):
        formula_string = "".join("{}{}".format(element, amount != 1 and amount or "")
                                for element, amount 
                                in sorted(self.composition.items()) 
                                if element != "charge")
        if self.composition["charge"]:
            if self.composition["charge"] == 1:
                formula_string += "+"
            elif self.composition["charge"] == -1:
                formula_string += "-"
            else:
                formula_string += "{:+d}".format(self.composition["charge"])
        return formula_string

    def __str__(self):
        return self.name

def load_chems_dict(path):
    with open(path, "r") as chems_fh:
        next(chems_fh)
        chems_dict = dict()
        for line in chems_fh:
            if not line.startswith("#"):
                name, formula, dGf0, source = line.rstrip().split(",")
                chems_dict[name] = Chem(name, formula, float(dGf0))
    return chems_dict
