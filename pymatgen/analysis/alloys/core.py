"""
This defines an `AlloySystem` class which is itself made up of `AlloyPairs`, for
example the system Al-Ga-In-N contains pairs Al-Ga-N, Al-In-N, Ga-In-N. All
entries in an AlloySystem must be commensurate with each other.

A `FormulaAlloyPair` class contains `AlloyPairs` which have formation energies
known to estimate which AlloyPair is stable for a given composition.
TODO: A `FormulaAlloySystem` is defined consisting of `FormulaAlloyPair` and specifies
the full space accessible for a given composition.
"""

from dataclasses import dataclass, field

import hashlib
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px

import plotly.graph_objects as go
from monty.serialization import loadfn
from pathlib import Path
from itertools import groupby, chain
from monty.json import MSONable
from scipy.spatial.qhull import HalfspaceIntersection
from shapely.geometry import MultiPoint
from typing import List, Tuple, Optional, Dict, Literal, Any, Set
from scipy.constants import c, h, elementary_charge

from pymatgen.core.composition import Species, Composition
from pymatgen.analysis.structure_matcher import ElementComparator
from pymatgen.core.structure import Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.transformations.standard_transformations import (
    AutoOxiStateDecorationTransformation,
    ConventionalCellTransformation,
)
from pymatgen.util.string import unicodeify

from pymatgen.analysis.alloys.rgb import rgb

# structure matching parameters
LTOL: float = 0.2
STOL: float = 0.3
ANGLE_TOL: float = 5

# Provided as convenience based on Materials Project data
KNOWN_ANON_FORMULAS = loadfn(Path(__file__).parent / "anon_formulas.json")


def ev_to_rgb(ev: float) -> Tuple[float, float, float]:
    """
    Function to convert energy (eV) into a colour for illustrative
    purposes. Beware using this for rigorous applications since
    digital representations of colour are tricky!

    See rgb.py for underlying representation.

    :param ev: Photon energy in electronvolts.
    :return: A (red, green, blue) tuple in the range 0-255
    """
    nm = ((10 ** 9) * c * h) / (elementary_charge * ev)
    return tuple(rgb(nm))


class InvalidAlloy(ValueError):
    """
    Exception raised for any invalid alloy due to the alloy itself
    being physically invalid. Exception is not raised because inputs
    are entered incorrectly.
    """

    pass


# These supported properties are given as type hints only,
# technically the class will work with any property name.
SupportedProperties = Literal[
    "energy_above_hull",
    "formation_energy_per_atom",
    "band_gap",
    "is_gap_direct",
    "m_n",
    "m_p",
    "theoretical",
    "is_metal",
]


@dataclass
class AlloyMember(MSONable):
    """
    Light-weight class for specifying information about a member of an
    AlloyPair or AlloySystem.
    """

    id_: str
    db: str
    composition: Composition
    x: float
    is_ordered: bool


@dataclass
class AlloyPair(MSONable):
    """
    Data class for creating an alloy pair with commensurate pymatgen
    Structure end-points and one changing atomic variable, where A and
    B are end-point materials.

    Attributes:
        formula_a (str): Reduced chemical formula for end-point material A
        formula_b (str): Reduced chemical formula for end-point material B
        structure_a (Structure): Pymatgen Structure for end-point material A
        structure_b (Structure): Pymatgen Structure for end-point material B
        id_a (str): Unique identifier for end-point material A, e.g. Materials Project material_id
        id_b (str): Unique identifier for end-point material B, e.g. Materials Project material_id
        alloying_element_a (str): Element to be alloyed in end-point material A
        alloying_element_b (str): Element to be alloyed in end-point material B
        alloying_species_a (Optional[str]): If oxidation state detected, species to
            be alloyed in end-point material A
        alloying_species_b (Optional[str]): If oxidation state detected, species to
            be alloyed in end-point material B
        anions_a (List[str]): Anions with oxidation state in end-point material A
        anions_b (List[str]): Anions with oxidation state in end-point material B
        cations_a (List[str]): Cations with oxidation state in end-point material A
        cations_b (List[str]): Cations with oxidation state in end-point material B
        observer_elements (List[str]): ...
        observer_species (List[str]): ...
        lattice_parameters_a (List[float]): Conventional lattice parameters,
            formatted as [a, b, c, alpha, beta, gamma], for end-point material A
        lattice_parameters_b (List[float]): Conventional lattice parameters,
            formatted as [a, b, c, alpha, beta, gamma], for end-point material B
        properties_a (dict): Materials properties of end-point material A
            that may or may not be populated. Suggested keys are "e_above_hull",
            "formation_energy", "band_gap, "is_gap_direct", "m_n", "m_p"
        properties_b (dict): Materials properties of end-point material A
            that may or may not be populated. Suggested keys are "e_above_hull",
            "formation_energy", "band_gap, "is_gap_direct", "m_n", "m_p"
        volume_cube_root_a (float): Cube root of the volume of the primitive
            unit cell for end-point material A, in Angstroms
        volume_cube_root_b (float): Cube root of the volume of the primitive
            unit cell for end-point material B, in Angstroms
        spacegroup_intl_number_a (int): International space group number of end-point
            material A
        spacegroup_intl_number_b (int): International space group number of end-point
            material B
        pair_id (str): A unique identifier for this alloy pair
        pair_formula (str): A human-readable identifier for this alloy pair
        alloy_oxidation_state (Optional[int]): If set, will be the common oxidation state for
            alloying elements in both end-points.
        isoelectronic (Optional[bool]): If set, will give whether the alloying elements
            are expected to be isoelectronic using their oxidation state. This is a
            simplistic method calculated based on the alloying elements' groups.
        anonymous_formula (str): Anonymous formula for both end-points (must be the
            same for this class which does not consider incommensurate alloys)
        nelements (int): Number of elements in end-point structure
    """

    # some fields are not shown in the __repr__ for brevity
    # some fields are not used in the __init__ since they can be generated deterministically

    formula_a: str
    formula_b: str
    structure_a: Structure = field(repr=False)
    structure_b: Structure = field(repr=False)
    id_a: str
    id_b: str
    chemsys: str
    alloying_element_a: str
    alloying_element_b: str
    alloying_species_a: Optional[str]
    alloying_species_b: Optional[str]
    observer_elements: List[str]
    observer_species: Optional[List[str]]
    anions_a: List[str]
    anions_b: List[str]
    cations_a: List[str]
    cations_b: List[str]
    lattice_parameters_a: List[float] = field(repr=False)
    lattice_parameters_b: List[float] = field(repr=False)
    volume_cube_root_a: float
    volume_cube_root_b: float
    properties_a: Dict[SupportedProperties, Any] = field(repr=False)
    properties_b: Dict[SupportedProperties, Any] = field(repr=False)
    spacegroup_intl_number_a: int
    spacegroup_intl_number_b: int
    pair_id: str = field(repr=False)
    pair_formula: str = field(repr=False)
    alloy_oxidation_state: Optional[int] = field(repr=False)
    isoelectronic: Optional[bool]
    anonymous_formula: str = field(repr=False)
    nelements: int = field(repr=False)
    members: List[AlloyMember] = field(default_factory=list, repr=False)

    def __post_init__(self):
        """
        Does check that formula are sorted correctly for our alloy convention.
        """

        if self.formula_a > self.formula_b:
            raise ValueError("By convention, formula_a and formula_b must be sorted by alphabetical order.")

    @property
    def alloy_formula(self):
        return unicodeify(self.formula_a).replace(
            self.alloying_element_a, f"({self.alloying_element_b}ₓ{self.alloying_element_a}₁₋ₓ)",
        )

    def __str__(self):
        return f"AlloyPair {self.alloy_formula}"

    @staticmethod
    def _get_anions_and_cations(structure: Structure, attempt_to_guess: bool = False) -> Tuple[List[str], List[str]]:
        """
        Method to get anions and cations from a structure as strings with oxidation
        state included.

        :param structure: Structure, ideally already oxidation-state decorated
        :param attempt_to_guess: if True, will attempt to guess oxidation states if not
        already specified,
        :return:
        """

        if attempt_to_guess:

            structure = structure.copy()

            trans = AutoOxiStateDecorationTransformation()

            # decorate with oxidation states, prefer bond valence if it works
            try:
                structure = trans.apply_transformation(structure)
            except ValueError:
                try:
                    structure.add_oxidation_state_by_guess()
                except ValueError:
                    # if both methods fail
                    return [], []

        anion_list = []
        cation_list = []

        for sp in structure.types_of_species:
            # check that sp is species not Element *and* it has a non-None oxi_state
            if hasattr(sp, "oxi_state") and sp.oxi_state:
                if sp.oxi_state > 0:
                    cation_list.append(str(sp))
                if sp.oxi_state < 0:
                    anion_list.append(str(sp))

        anion_list = sorted(anion_list)
        cation_list = sorted(cation_list)

        return anion_list, cation_list

    @classmethod
    def from_structures(
        cls,
        structures: Tuple[Structure, Structure],
        structures_with_oxidation_states: Tuple[Structure, Structure],
        ids: Tuple[str, str],
        properties: Optional[Tuple[Dict[SupportedProperties, Any], Dict[SupportedProperties, Any]]] = None,
        ltol: float = LTOL,
        stol: float = STOL,
        angle_tol: float = ANGLE_TOL,
    ) -> "AlloyPair":
        """
        ...

        :param structures:
        :param structures_with_oxidation_states:
        :param ids:
        :param properties:
        :param ltol:
        :param stol:
        :param angle_tol:
        :return:
        """

        if len(structures) != len(ids) != 2:
            raise InvalidAlloy("An alloy system must have two and only two end-points.")

        if properties and len(properties) != 2:
            raise ValueError

        if not properties:
            properties = ({}, {})

        formulas_and_structures = [
            (s.composition.reduced_formula, id_, s, s_oxi, property)
            for id_, s, s_oxi, property in zip(ids, structures, structures_with_oxidation_states, properties)
        ]

        # ensure A is always the same regardless of order of input ids
        formulas_and_structures = sorted(formulas_and_structures)
        (
            (formula_a, id_a, structure_a, structure_oxi_a, properties_a),
            (formula_b, id_b, structure_b, structure_oxi_b, properties_b),
        ) = formulas_and_structures

        (alloying_element_a, alloying_element_b,) = cls._get_alloying_elements_for_commensurate_structures(
            structure_a, structure_b, ltol=ltol, stol=stol, angle_tol=angle_tol
        )
        anions_a, cations_a = cls._get_anions_and_cations(structure_oxi_a)
        anions_b, cations_b = cls._get_anions_and_cations(structure_oxi_b)

        (
            alloy_oxidation_state,
            alloying_species_a,
            alloying_species_b,
            oxi_state_a,
            oxi_state_b,
            isoelectronic,
        ) = cls._get_oxi_state_info(anions_a, cations_a, anions_b, cations_b, alloying_element_a, alloying_element_b,)

        conv = ConventionalCellTransformation()
        conv_structure_a = conv.apply_transformation(structure_a)
        conv_structure_b = conv.apply_transformation(structure_b)
        lattice_params_a = conv_structure_a.lattice.parameters
        lattice_params_b = conv_structure_b.lattice.parameters

        all_elements = set()
        for formula in (formula_a, formula_b):
            for el in Composition(formula).elements:
                all_elements.add(str(el))
        chemsys = "-".join(sorted(all_elements))

        all_species = set(structure_oxi_a.types_of_species) | set(structure_oxi_b.types_of_species)
        all_species = {str(sp) for sp in all_species if isinstance(sp, Species)}
        observer_species = (
            list(all_species - {alloying_species_a, alloying_species_b})
            if (alloying_species_a and alloying_species_b)
            else []
        )
        observer_elements = list(all_elements - {alloying_element_a, alloying_element_b})

        system = cls(
            id_a=id_a,
            id_b=id_b,
            formula_a=formula_a,
            formula_b=formula_b,
            chemsys=chemsys,
            structure_a=structure_a,
            structure_b=structure_b,
            alloying_element_a=alloying_element_a,
            alloying_element_b=alloying_element_b,
            alloying_species_a=alloying_species_a,
            alloying_species_b=alloying_species_b,
            anions_a=anions_a,
            anions_b=anions_b,
            cations_a=cations_a,
            cations_b=cations_b,
            observer_elements=observer_elements,
            observer_species=observer_species,
            lattice_parameters_a=lattice_params_a,
            lattice_parameters_b=lattice_params_b,
            volume_cube_root_a=structure_a.get_primitive_structure().volume ** (1 / 3),
            volume_cube_root_b=structure_b.get_primitive_structure().volume ** (1 / 3),
            properties_a=properties_a,
            properties_b=properties_b,
            spacegroup_intl_number_a=structure_a.get_space_group_info()[1],
            spacegroup_intl_number_b=structure_b.get_space_group_info()[1],
            pair_id=f"{id_a}_{id_b}",
            pair_formula=f"{formula_a}_{formula_b}",
            alloy_oxidation_state=alloy_oxidation_state,
            isoelectronic=isoelectronic,
            anonymous_formula=structure_a.composition.anonymized_formula,
            nelements=len(structure_a.composition.element_composition.elements),
            members=[],
        )

        return system

    @staticmethod
    def _get_oxi_state_info(
        anions_a: List[str],
        cations_a: List[str],
        anions_b: List[str],
        cations_b: List[str],
        alloying_element_a: str,
        alloying_element_b: str,
    ) -> Tuple[
        Optional[float], Optional[str], Optional[str], Optional[float], Optional[float], Optional[bool],
    ]:
        """
        Get information about what oxidation states are present in the alloy, such
        as if the alloy is isoelectronic or not.
        """

        # determine the oxidation state of the alloying element if detected and
        # whether these oxidation states are the same
        alloy_oxidation_state = None
        alloying_species_a = None
        alloying_species_b = None
        oxi_state_a = None
        oxi_state_b = None
        isoelectronic = None

        if (anions_a or cations_a) and (anions_b or cations_b):

            ions_a = [Species.from_string(sp) for sp in anions_a] + [Species.from_string(sp) for sp in cations_a]
            elements_a = [str(sp.element) for sp in ions_a]

            ions_b = [Species.from_string(sp) for sp in anions_b] + [Species.from_string(sp) for sp in cations_b]
            elements_b = [str(sp.element) for sp in ions_b]

            # check for rare situation where maybe multiple oxidation states defined for the same element
            # and it's ambiguous what the true oxidation state of the alloying element is
            if (alloying_element_a in elements_a) and (elements_a.count(alloying_element_a) == 1):
                index_a = elements_a.index(alloying_element_a)
                alloying_species_a = str(ions_a[index_a])
                oxi_state_a = ions_a[index_a].oxi_state

            if (alloying_element_b in elements_b) and (elements_b.count(alloying_element_b) == 1):
                index_b = elements_b.index(alloying_element_b)
                alloying_species_b = str(ions_b[index_b])
                oxi_state_b = ions_b[index_b].oxi_state

            if oxi_state_a == oxi_state_b:
                alloy_oxidation_state = oxi_state_a

            if alloying_species_a and alloying_species_b:
                # this is a very simplistic method based on periodic table group to see
                # if alloying elements are isoelectronic, should be check
                if (ions_a[index_a].element.group - oxi_state_a) - (ions_b[index_b].element.group - oxi_state_b) == 0:
                    isoelectronic = True
                else:
                    isoelectronic = False

        return (
            alloy_oxidation_state,
            alloying_species_a,
            alloying_species_b,
            oxi_state_a,
            oxi_state_b,
            isoelectronic,
        )

    def is_member(
        self, structure: Structure, ltol: float = LTOL, stol: float = STOL, angle_tol: float = ANGLE_TOL
    ) -> bool:
        """
        Check if a Structure could be a member of the AlloyPair.

        This method is necessarily a heuristic and can give
        false positives or negatives.

        If space groups match, it is assumed to be a member. If space groups do not match,
        a StructureMatcher comparison is performed.

        :param structure: Structure to check
        :return: True or False
        """

        # TODO: allow competing_pairs kwarg which may change the behavior of this method depending on if
        # it is known that other alloy pairs exist with this composition

        if self.chemsys != structure.composition.chemical_system:
            return False

        try:
            # can sometimes fail due to spglib returning None, unfortunately
            spacegroup_intl_number = structure.get_space_group_info()[1]
            if (self.spacegroup_intl_number_a == spacegroup_intl_number) or (self.spacegroup_intl_number_b) == (
                    spacegroup_intl_number
            ):
                # heuristic! may give false positives
                return True
        except TypeError:
            pass

        # create two test structures from input ABX: one of composition AX and one BX
        structure_a, structure_b = structure.copy(), structure.copy()
        structure_a.remove_oxidation_states()
        structure_b.remove_oxidation_states()

        structure_a.replace_species({self.alloying_element_b: self.alloying_element_a})
        structure_b.replace_species({self.alloying_element_a: self.alloying_element_b})

        if self.structure_a.matches(
            structure_a, ltol=ltol, stol=stol, angle_tol=angle_tol, comparator=ElementComparator()
        ):
            return True

        if self.structure_b.matches(
            structure_b, ltol=ltol, stol=stol, angle_tol=angle_tol, comparator=ElementComparator()
        ):
            return True

        return False

    def get_x(self, composition: Composition) -> float:
        """
        Calculate the position of a composition along
        an input line.

        :param composition: input composition
        :return: x
        """
        c = composition.element_composition
        if (self.alloying_element_a not in c) or (self.alloying_element_b not in c):
            raise ValueError("Provided composition does not contain required alloying elements.")
        return c[self.alloying_element_a] / (c[self.alloying_element_a] + c[self.alloying_element_b])

    @staticmethod
    def _get_alloying_elements_for_commensurate_structures(
        structure_a: Structure,
        structure_b: Structure,
        ltol: float = LTOL,
        stol: float = STOL,
        angle_tol: float = ANGLE_TOL,
    ) -> Tuple[str, str]:
        """
        Run a series of checks to ensure alloys structures are commensurate to the first order, and
        that only one element differs between structures.
        """

        # check equivalent anonymous formula
        if structure_a.composition.anonymized_formula != structure_b.composition.anonymized_formula:
            raise InvalidAlloy("Alloys are not commensurate")

        # check one varying element making sure to ignore oxidation state if present
        types_of_species_a = [sp.element if isinstance(sp, Species) else sp for sp in structure_a.types_of_species]
        types_of_species_b = [sp.element if isinstance(sp, Species) else sp for sp in structure_b.types_of_species]

        element_a_set = set(types_of_species_a).difference(structure_b.types_of_species)
        if len(element_a_set) != 1:
            raise InvalidAlloy(
                f"Alloy end-points must have one and only one changing element ({element_a_set}) "
                f"from {types_of_species_a} and {types_of_species_b}"
            )
        alloying_element_a = str(element_a_set.pop())
        alloying_element_b = str(set(types_of_species_b).difference(types_of_species_a).pop())

        structure_b_copy = structure_b.copy()
        structure_b_copy.replace_species({alloying_element_b: alloying_element_a})

        if not structure_a.matches(
            structure_b_copy, ltol=ltol, stol=stol, angle_tol=angle_tol, comparator=ElementComparator(),
        ):
            raise InvalidAlloy("End-point structures do not match")

        return alloying_element_a, alloying_element_b

    def as_records(self, fields=None):
        """
        Convert to a record for a pandas DataFrame to remove _a, _b subscripts for easier plotting.
        """

        record_a = {"is": "a", "x": 0}
        record_b = {"is": "b", "x": 1}

        # quick way to get all attributes for _a and _b
        # only works because we use this naming convention
        for attr in self.__dir__():
            if not attr.startswith("_"):
                if attr.endswith("_a"):
                    record_a[attr[:-2]] = getattr(self, attr)
                elif attr.endswith("_b"):
                    record_b[attr[:-2]] = getattr(self, attr)
                else:
                    record_a[attr] = getattr(self, attr)
                    record_b[attr] = getattr(self, attr)

        for attr, value in self.properties_a.items():
            record_a[attr] = value
        for attr, value in self.properties_b.items():
            record_b[attr] = value

        if fields:
            record_a = {k: v for k, v in record_a.items() if k in fields}
            record_b = {k: v for k, v in record_b.items() if k in fields}

        return [record_a, record_b]

    def search_dict(self) -> dict:
        """
        Additional fields from which to build search indices.

        :return: dict
        """

        search = dict()

        # for search, we want to search on min and max for float values
        # and create lists for string values
        numerical_attrs = set()
        other_attrs = set()
        # quick way to get all attributes for _a and _b
        # only works because we use this naming convention
        for attr in self.__dir__():
            if not attr.startswith("_"):
                if attr.endswith("_a") or attr.endswith("_b"):
                    sample_attr = getattr(self, attr)
                    if isinstance(sample_attr, float) or isinstance(sample_attr, int):
                        numerical_attrs.add(attr[:-2])
                    elif isinstance(sample_attr, str):
                        other_attrs.add(attr[:-2])

        for attr in numerical_attrs:
            attr_range = [getattr(self, f"{attr}_a"), getattr(self, f"{attr}_b")]
            attr_range = [r for r in attr_range if r is not None]
            if attr_range:
                search[attr] = {"min": min(attr_range), "max": max(attr_range)}

        for attr in other_attrs:
            fields = [getattr(self, f"{attr}_a"), getattr(self, f"{attr}_b")]
            fields = [f for f in fields if f is not None]
            if fields:
                search[attr] = fields

        # and do the same for properties

        numerical_properties = set()
        other_properties = set()
        for prop, value_a in self.properties_a.items():
            if isinstance(value_a, float) or isinstance(value_a, int):
                numerical_properties.add(prop)
            elif isinstance(value_a, str):
                other_properties.add(prop)
        for prop, value_b in self.properties_b.items():
            if isinstance(value_b, float) or isinstance(value_b, int):
                numerical_properties.add(prop)
            elif isinstance(value_b, str):
                other_properties.add(prop)

        for prop in numerical_properties:
            prop_range = [self.properties_a.get(prop), self.properties_b.get(prop)]
            prop_range = [r for r in prop_range if r is not None]
            if prop_range:
                search[prop] = {"min": min(prop_range), "max": max(prop_range)}

        for prop in other_properties:
            fields = [self.properties_a.get(prop), self.properties_b.get(prop)]
            fields = [f for f in fields if f is not None]
            if fields:
                search[prop] = fields

        return search


@dataclass
class AlloySystem(MSONable):
    """
    An alloy system defined by a group of AlloyPairs.

    Attributes:
        ids (List[str]): A flat list of all ids in the
            alloy system
        pair_ids (List[str]): A list of id pairs defining
            the given alloy pair (where ids are underscore
            delimited and given in lexicographical order of their
            reduced formula, this format matches the format
            from pair_id given in AlloyPair so is easy to
            query in a database)
        alloy_pairs (List[AlloyPair]): A list of the AlloyPairs
            belonging to this system
        alloy_id (str): A unique identifier for this alloy system
    """

    ids: Set[str]
    pair_ids: Set[str] = field(init=False)
    alloy_pairs: List[AlloyPair] = field(repr=False)
    alloy_id: str = field(init=False)
    n_pairs: int = field(init=False)
    chemsys: str = field(init=False)
    chemsys_size: int = field(init=False, repr=False)
    # sets of alloying_elements etc?
    # property_ranges: Dict[SupportedProperties, Tuple[float, float]] = field(init=False)

    def __post_init__(self):
        # TODO: construct property_ranges here

        self.alloy_id = hashlib.sha256("_".join(sorted(self.ids)).encode("utf-8")).hexdigest()[:6]

        self.n_pairs = len(self.alloy_pairs)
        self.pair_ids = set()

        chemsys = set()
        for pair in self.alloy_pairs:
            self.pair_ids.add(pair.pair_id)
            for el in pair.chemsys.split("-"):
                chemsys.add(el)
        self.chemsys_size = len(chemsys)
        self.chemsys = "-".join(sorted(chemsys))

    @staticmethod
    def systems_from_pairs(alloy_pairs: List[AlloyPair]):

        g = nx.Graph()

        for pair in alloy_pairs:
            g.add_edge(pair.id_a, pair.id_b)

        subgraphs = nx.connected_components(g)

        def get_pairs_from_mpid_bunch(mpid_bunch):
            return [pair for pair in alloy_pairs if ((pair.id_a in mpid_bunch) or (pair.id_b in mpid_bunch))]

        systems = [
            AlloySystem(ids=set(subgraph), alloy_pairs=get_pairs_from_mpid_bunch(subgraph)) for subgraph in subgraphs
        ]

        return systems

    def get_property(self, mpid, prop):
        for pair in self.alloy_pairs:
            if pair.id_a == mpid:
                if hasattr(pair, f"{prop}_a"):
                    return getattr(pair, f"{prop}_a")
                elif prop in pair.properties_a:
                    return pair.properties_a[prop]
                else:
                    return getattr(pair, f"{prop}")
            elif pair.id_b == mpid:
                if hasattr(pair, f"{prop}_b"):
                    return getattr(pair, f"{prop}_b")
                elif prop in pair.properties_b:
                    return pair.properties_b[prop]
                else:
                    return getattr(pair, f"{prop}")

    def filter_by(self, prop_constraint) -> "AlloySystem":
        # for plotting?
        raise NotImplementedError

    def get_verts(self, x_prop="volume_cube_root", y_prop="band_gap"):
        pass

    def get_convex_hull_and_centroid(self, x_prop="volume_cube_root", y_prop="band_gap"):

        points = [(self.get_property(mpid, x_prop), self.get_property(mpid, y_prop)) for mpid in self.ids]

        if len(points) < 3:
            return None, None, None

        hull = MultiPoint(points).convex_hull

        return list(hull.exterior.coords), list(hull.centroid.coords)[0], hull.area

    def get_hull_trace_and_area(
        self, x_prop="volume_cube_root", y_prop="band_gap", colour=(0, 0, 0), opacity=0.2, colour_by_centroid=False
    ):

        hull, centroid, area = self.get_convex_hull_and_centroid(x_prop, y_prop)

        if not hull:
            return None, None

        if y_prop == "band_gap" and colour_by_centroid:
            try:
                colour = ev_to_rgb(centroid[1])
            except ValueError:  # wavelength out of range
                pass

        trace = go.Scatter(
            x=[p[0] for p in hull],
            y=[p[1] for p in hull],
            fill="toself",
            fillcolor=f"rgba({colour[0]},{colour[1]},{colour[2]},{opacity})",
            hoverinfo="skip",
            mode="none",
            showlegend=False,
        )

        return trace, area

    def plot(
        self,
        x_prop="volume_cube_root",
        y_prop="band_gap",
        symbol="theoretical",
        column_mapping=None,
        plotly_pxline_kwargs=None,
        plotly_pxscatter_kwargs=None,
    ):

        data = []

        # used to set human-readable column names
        column_mapping = column_mapping or {x_prop: x_prop, y_prop: y_prop, symbol: symbol}

        for pair in self.alloy_pairs:
            data.append(
                {
                    column_mapping[x_prop]: getattr(pair, f"{x_prop}_a", None) or pair.properties_a.get(x_prop),
                    column_mapping[y_prop]: getattr(pair, f"{y_prop}_a", None) or pair.properties_a.get(y_prop),
                    column_mapping[symbol]: getattr(pair, f"{symbol}_a", None) or pair.properties_a.get(symbol),
                    "mpid": pair.id_a,
                    "formula": pair.formula_a,
                    "pair_id": pair.pair_id,
                }
            )
            data.append(
                {
                    column_mapping[x_prop]: getattr(pair, f"{x_prop}_b", None) or pair.properties_b.get(x_prop),
                    column_mapping[y_prop]: getattr(pair, f"{y_prop}_b", None) or pair.properties_b.get(y_prop),
                    column_mapping[symbol]: getattr(pair, f"{symbol}_b", None) or pair.properties_b.get(symbol),
                    "mpid": pair.id_b,
                    "formula": pair.formula_b,
                    "pair_id": pair.pair_id,
                }
            )

        df = pd.DataFrame(data)

        pxline_kwargs = dict(
            line_group="pair_id",
            text="formula",
            labels="formula",
            hover_data=["formula", "mpid"],
            title=f"Alloy system: {self.alloy_id}",
            markers=False,
        )
        plotly_pxline_kwargs = plotly_pxline_kwargs or {}
        pxline_kwargs.update(plotly_pxline_kwargs)

        fig = px.line(df, column_mapping[x_prop], column_mapping[y_prop], **pxline_kwargs)
        fig.update_traces(textposition="top center")

        if symbol:

            pxscatter_kwargs = dict(symbol=column_mapping[symbol])
            plotly_pxscatter_kwargs = plotly_pxscatter_kwargs or {}
            pxscatter_kwargs.update(plotly_pxscatter_kwargs)

            scatter_fig = px.scatter(df, column_mapping[x_prop], column_mapping[y_prop], **pxscatter_kwargs)
            scatter_fig.update_traces(marker={"size": 12})
            for trace in scatter_fig.data:
                fig.add_trace(trace)

        if y_prop == "band_gap":
            for ev in np.arange(1.6, 3.1, 0.05):
                fillcolor = "rgb({},{},{})".format(*ev_to_rgb((ev + ev + 0.1) / 2))
                fig.add_hrect(y0=ev, y1=ev + 0.05, fillcolor=fillcolor, layer="below", line_width=0)

        return fig

    def __len__(self):
        return len(self.pair_ids)

    def as_dict_mongo(self):
        # do not store AlloyPairs?
        # add number of components
        # add search dict
        raise NotImplementedError


def combine_systems(alloy_systems: List[AlloySystem]) -> List[AlloySystem]:
    pass


@dataclass
class FormulaAlloyPair(MSONable):

    segments: List
    pairs: List[AlloyPair]

    @classmethod
    def from_pairs(cls, pairs: List[AlloyPair]) -> List["FormulaAlloyPair"]:
        """
        Gets halfspace segments and pairs for a set of AlloyPair objects

        Args:
            pairs

        """

        pairs = sorted(pairs, key=lambda pair: pair.pair_formulas)
        alloy_pair_groups = groupby(pairs, key=lambda pair: pair.pair_formulas)

        formula_alloy_pairs = []

        for pair_formula, alloy_pair_group in alloy_pair_groups:
            pairs = list(alloy_pair_group)

            alloy_pair_df = pd.DataFrame(chain.from_iterable(pair.as_records() for pair in pairs))

            hull_df, segments = cls._get_hull(alloy_pair_df)

            formula_alloy_pairs.append(cls(pairs=pairs, segments=segments))

        return formula_alloy_pairs

    @staticmethod
    def _get_line_equation(point_1, point_2):

        A = point_1[1] - point_2[1]
        B = point_2[0] - point_1[0]
        C = point_1[0] * point_2[1] - point_2[0] * point_1[1]
        return A, B, C

    @staticmethod
    def _get_hull(alloy_pair_df):

        pairs_ids = alloy_pair_df["pair_ids"].unique()

        lines = []
        # construct so that the order of lines matches the order of pairs_ids
        for pair in pairs_ids:
            df_pair = alloy_pair_df[alloy_pair_df["pair_ids"] == pair]
            a_1 = (
                df_pair[df_pair["is"] == "a"]["x"].values[0],
                df_pair[df_pair["is"] == "a"]["e_above_hull"].values[0],
            )
            b_1 = (
                df_pair[df_pair["is"] == "b"]["x"].values[0],
                df_pair[df_pair["is"] == "b"]["e_above_hull"].values[0],
            )
            lines.append(FormulaAlloyPair._get_line_equation(a_1, b_1))
        lines += [[-1, 0, 0], [1, 0, -1]]  # add end-points

        # define halfspaces
        halfspaces = np.array(lines)

        # define a point in alloy space with unphysically low energy
        feasible_point = np.array([0.5, -1])
        hs = HalfspaceIntersection(halfspaces, feasible_point)

        a_min = alloy_pair_df[alloy_pair_df["is"] == "a"].sort_values("e_above_hull").iloc[0]
        a_endpoint = (a_min["x"], a_min["e_above_hull"])
        b_min = alloy_pair_df[alloy_pair_df["is"] == "b"].sort_values("e_above_hull").iloc[0]
        b_endpoint = (b_min["x"], b_min["e_above_hull"])

        points = [a_endpoint, b_endpoint]
        for i in hs.intersections:
            if (i[0] > 0) & (i[0] < 1):
                points.append(i)

        hull_df = pd.DataFrame(points, columns=["x", "e_above_hull"]).sort_values("x")
        hull_df = hull_df.reset_index().drop(columns="index")

        # find hull phases
        intersection_to_pairs = {}
        for idx, (x, y) in enumerate(hs.intersections):
            if not np.isnan(x):
                vertex_1, vertex_2 = hs.dual_facets[idx]
                pairs = []
                if vertex_1 < len(pairs_ids):
                    pairs.append(pairs_ids[vertex_1])
                if vertex_2 < len(pairs_ids):
                    pairs.append(pairs_ids[vertex_2])
                intersection_to_pairs[x] = pairs

        # sort by x value
        intersection_to_pairs = sorted(intersection_to_pairs.items())

        # find segments corresponding to specific alloy pair
        segments = []
        for i in range(len(intersection_to_pairs) - 1):
            x_segment_start = intersection_to_pairs[i][0]
            x_segment_end = intersection_to_pairs[i + 1][0]
            mpid_pair = set(intersection_to_pairs[i][1]).intersection(intersection_to_pairs[i + 1][1]).pop()
            segments.append(
                {"x_segment_start": x_segment_start, "x_segment_end": x_segment_end, "mpid_pair": mpid_pair}
            )

        return hull_df, segments

    @staticmethod
    def _get_alloy_polymorphs_from_mp(df_alloy):

        formula_a = list(df_alloy.loc[df_alloy["is"] == "a", "formula"])[0]
        formula_b = list(df_alloy.loc[df_alloy["is"] == "b", "formula"])[0]
        with MPRester() as mpr:
            polymorphs = mpr.query(
                {"pretty_formula": {"$in": [formula_a, formula_b]}},
                ["pretty_formula", "task_id", "spacegroup.symbol", "e_above_hull", "theoretical"],
            )
        df_polymorphs = pd.DataFrame(polymorphs)
        alloy_map = {formula_a: 0, formula_b: 1}
        df_polymorphs["x"] = [alloy_map[iz] for iz in df_polymorphs["pretty_formula"]]
        return df_polymorphs

    def plot(self, supplement_with_mp=True, w=1000, h=400):

        fields = [
            "x",
            "is",
            "formula",
            "mpid",
            "e_above_hull",
            "formation_energy",
            "spacegroup_intl_number",
            "pair_ids",
            "pair_formulas",
            "alloy_formula",
        ]
        df = pd.DataFrame(chain.from_iterable(pair.as_records(fields=fields) for pair in self.pairs))

        hull_df, _ = FormulaAlloyPair._get_hull(df)

        fig = px.line(
            df,
            x="x",
            y="e_above_hull",
            color="pair_ids",
            line_group="pair_ids",
            line_dash_sequence=["dot"],
            hover_data=fields,
        )
        fig.add_scatter(
            x=hull_df["x"],
            y=hull_df["e_above_hull"],
            mode="lines + markers",
            name="hull",
            opacity=0.5,
            marker_color="black",
            marker_size=10,
            hoverinfo=None,
        )

        if supplement_with_mp:
            df_polymorphs = FormulaAlloyPair._get_alloy_polymorphs_from_mp(df)
            fig.add_scatter(
                x=df_polymorphs["x"],
                y=df_polymorphs["e_above_hull"],
                mode="markers",
                marker_color="black",
                marker_size=5,
                name="MP polymorphs",
                hoverinfo="all",
            )

        fig.update_layout(title=df["alloy_formula"][0], legend_title_text="")
        fig.update_yaxes(title="E<sub>hull</sub> (eV/atom)")
        fig.update_layout(
            autosize=False,
            xaxis=dict(showgrid=False, mirror=True),
            yaxis=dict(showgrid=False, mirror=True),
            height=h,
            width=w,
            # bargap=0,
            hovermode="closest",
            template="simple_white",
            font=dict(family="Helvetica", size=16, color="black"),
        )

        return fig

    def __str__(self):
        return f"FormulaAlloyPair {self.pairs[0].alloy_formula}"
