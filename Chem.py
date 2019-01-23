import re

class Chem:
    '''Chemical specie'''
    # Chem instances still have a "charge" attribute so formulaless chems
    # can be treated homogeneously with molecules when computing charge balance.
    # Their charge can be non-null so chems such as NAD+ can be described
    # correctly in this regard
    charge_pat = re.compile("([+-])([0-9]*)$")

    def __init__(self, name, charge=0):
        self.name = name
        self.composition = {}
        charge_string = self.__class__.charge_pat.findall(name)
        if charge_string:
            sign, charge = charge_string[0]
            self.composition["charge"] = int(sign + (charge or "1"))
        else:
            self.composition["charge"] = 0

    def copy(self):
        return Chem(self.name, self.charge)

    def __str__(self):
        return self.name

class Molecule(Chem):
    '''Chemical specie with a defined chemical formula'''
    element_pat = re.compile("([A-Z](?:[a-z]*))([0-9]*)")

    def __init__(self, name, formula, dGf0):
        super().__init__(name)
        self.dGf0 = dGf0
        for element, amount in self.__class__.element_pat.findall(formula):
            self.composition[element] = amount and int(amount) or 1
        # charge was already infered from the name in Chem.__init__
        # however it was infered from the `name` attribute, while
        # a molecule's charge should be infered from its formula
        charge_string = self.__class__.charge_pat.findall(formula)
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

    def copy(self):
        return Molecule(self.name, self.formula(), self.dGf0)

def load_chems_dict(path):
    with open(path, "r") as chems_fh:
        next(chems_fh)
        chems_dict = dict()
        for line in chems_fh:
            if not line.startswith("#"):
                name, formula, dGf0, source = line.rstrip().split(",")
                chems_dict[name] = Molecule(name, formula, float(dGf0))
    return chems_dict
