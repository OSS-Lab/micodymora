import re

class Chem:
    '''Chemical specie'''
    # Chem instances still have a "charge" attribute so formulaless chems
    # can be treated homogeneously with molecules when computing charge balance.
    # Their charge can be non-null so chems such as NAD+ can be described
    # correctly in this regard
    charge_pat = re.compile("([+-])([0-9]*)$")
    phase_pat = re.compile("\((aq|g|s)\)")

    def __init__(self, name, charge=0):
        self.name = name
        self.composition = {}
        charge_string = self.__class__.charge_pat.findall(name)
        if charge_string:
            sign, charge = charge_string[0]
            self.composition["charge"] = int(sign + (charge or "1"))
        else:
            self.composition["charge"] = 0
        phase_tag = self.__class__.phase_pat.findall(name)
        self.phase = phase_tag and phase_tag[0] or None

    def copy(self, name=None):
        return Chem(name or self.name, self.charge)

    def __str__(self):
        return self.name

class Molecule(Chem):
    '''Chemical specie with a defined chemical formula'''
    element_pat = re.compile("([A-Z](?:[a-z]*))([0-9]*)")

    def __init__(self, name, formula, dGf0, dHf0=None):
        super().__init__(name)
        self.dGf0 = dGf0
        self.dHf0 = dHf0
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

    def copy(self, name=None):
        return Molecule(name or self.name, self.formula(), self.dGf0, dHf0=self.dHf0)

def load_chems_dict(path, dHf0_data_path=None):
    dHf0_data = dict()
    if dHf0_data_path:
        with open(dHf0_data_path, "r") as dHf0_fh:
            next(dHf0_fh)
            for line in dHf0_fh:
                if not line.startswith("#"):
                    name, dHf0, source = line.rstrip().split(",")
                    if dHf0 == "unknown":
                        dHf0_data[name] = None
                    else:
                        dHf0_data[name] = float(dHf0)

    with open(path, "r") as chems_fh:
        next(chems_fh)
        chems_dict = dict()
        for line in chems_fh:
            if not line.startswith("#"):
                name, formula, dGf0, source = line.rstrip().split(",")
                chems_dict[name] = Molecule(name, formula, float(dGf0), dHf0=dHf0_data.get(name))
    return chems_dict
