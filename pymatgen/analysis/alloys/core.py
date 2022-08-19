"""
This defines an `AlloySystem` class which is itself made up of `AlloyPairs`, for
example the system Al-Ga-In-N contains pairs Al-Ga-N, Al-In-N, Ga-In-N. All
entries in an AlloySystem must be commensurate with each other.

A `FormulaAlloyPair` class contains `AlloyPairs` which have formation energies
known to estimate which AlloyPair is stable for a given composition.
"""

# TODO: A `FormulaAlloySystem` is defined consisting of `FormulaAlloyPair` and specifies
# the full space accessible for a given composition.

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
from plotly.subplots import make_subplots
from pymatgen.analysis.phase_diagram import PhaseDiagram
from scipy.spatial.qhull import HalfspaceIntersection
from shapely.geometry import MultiPoint
from typing import List, Tuple, Optional, Dict, Literal, Any, Set, Callable, Union
from scipy.constants import c, h, elementary_charge

from pymatgen.core.composition import Species, Composition
from pymatgen.analysis.structure_matcher import ElementComparator
from pymatgen.core.structure import Structure
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
        formula_a (str): Reduced chemical formula for end-point material A.
        formula_b (str): Reduced chemical formula for end-point material B.
        structure_a (Structure): Pymatgen Structure for end-point material A.
        structure_b (Structure): Pymatgen Structure for end-point material B.
        id_a (str): Unique identifier for end-point material A, e.g. Materials Project material_id.
        id_b (str): Unique identifier for end-point material B, e.g. Materials Project material_id.
        alloying_element_a (str): Element to be alloyed in end-point material A.
        alloying_element_b (str): Element to be alloyed in end-point material B.
        alloying_species_a (Optional[str]): If oxidation state detected, species to
            be alloyed in end-point material A.
        alloying_species_b (Optional[str]): If oxidation state detected, species to
            be alloyed in end-point material B.
        anions_a (List[str]): Anions with oxidation state in end-point material A.
        anions_b (List[str]): Anions with oxidation state in end-point material B.
        cations_a (List[str]): Cations with oxidation state in end-point material A.
        cations_b (List[str]): Cations with oxidation state in end-point material B.
        observer_elements (List[str]): Elements in end-point materials A and B that
            are not alloyed.
        observer_species (List[str]): If oxidation state detected, species in
            end-point materials A and B that are not alloyed.
        lattice_parameters_a (List[float]): Conventional lattice parameters,
            formatted as [a, b, c, alpha, beta, gamma], for end-point material A.
        lattice_parameters_b (List[float]): Conventional lattice parameters,
            formatted as [a, b, c, alpha, beta, gamma], for end-point material B.
        properties_a (dict): Materials properties of end-point material A
            that may or may not be populated. Suggested keys are "energy_above_hull",
            "formation_energy_per_atom", "band_gap, "is_gap_direct", "m_n", "m_p"
        properties_b (dict): Materials properties of end-point material A
            that may or may not be populated. Suggested keys are "energy_above_hull",
            "formation_energy_per_atom", "band_gap, "is_gap_direct", "m_n", "m_p"
        volume_cube_root_a (float): Cube root of the volume of the primitive
            unit cell for end-point material A, in Angstroms.
        volume_cube_root_b (float): Cube root of the volume of the primitive
            unit cell for end-point material B, in Angstroms.
        spacegroup_intl_number_a (int): International space group number of end-point
            material A.
        spacegroup_intl_number_b (int): International space group number of end-point
            material B.
        pair_id (str): A unique identifier for this alloy pair.
        pair_formula (str): A human-readable identifier for this alloy pair.
        alloy_oxidation_state (Optional[int]): If set, will be the common oxidation state for
            alloying elements in both end-points.
        isoelectronic (Optional[bool]): If set, will give whether the alloying elements
            are expected to be isoelectronic using their oxidation state. This is a
            simplistic method calculated based on the alloying elements' groups.
        anonymous_formula (str): Anonymous formula for both end-points (must be the
            same for this class which does not consider incommensurate alloys).
        nelements (int): Number of elements in end-point structure.
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
        Perform check that formulas are sorted correctly for our alloy convention.
        """

        if self.formula_a > self.formula_b:
            raise ValueError("By convention, formula_a and formula_b must be sorted by alphabetical order.")

    @property
    def alloy_formula(self) -> str:
        """
        :return: Formatted alloy formula (e.g. AₓB₁₋ₓC).
        """
        return unicodeify(self.formula_a).replace(
            self.alloying_element_a, f"({self.alloying_element_b}ₓ{self.alloying_element_a}₁₋ₓ)",
        )

    @staticmethod
    def get_alloy_formula_from_formulas(formula_a: str, alloying_element_a: str, alloying_element_b: str) -> str:
        """
        Method to get alloy formula (e.g. AₓB₁₋ₓC) as a function of formulas of two alloying compounds.

        :param formula_a: Reduced chemical formula for end-point material A.
        :param alloying_element_a: Element to be alloyed in end-point material A.
        :param alloying_element_b: Element to be alloyed in end-point material B.
        :return: Formatted alloy formula (e.g. AₓB₁₋ₓC).
        """
        return unicodeify(formula_a).replace(alloying_element_a, f"({alloying_element_b}ₓ{alloying_element_a}₁₋ₓ)", )

    def __str__(self):
        return f"AlloyPair {self.alloy_formula}"

    @staticmethod
    def _get_anions_and_cations(structure: Structure, attempt_to_guess: bool = False) -> Tuple[List[str], List[str]]:
        """
        Method to get anions and cations from a structure as strings with oxidation
        state included.

        :param structure: Structure, ideally already oxidation-state decorated.
        :param attempt_to_guess: If True, will attempt to guess oxidation states if not
        already specified.
        :return: Anions and cations with oxidation state in input structure.
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
        Function to construct AlloyPair class.

        :param structures: Pair of pymatgen Structure objects.
        :param structures_with_oxidation_states: Pair of pymatgen Structure objects decorated
            with oxidation state.
        :param ids: Pair of unique identifiers, e.g. Materials Project material_id.
        :param properties: Pair of materials properties that may or may not be populated.
            Suggested keys are "e_above_hull", "formation_energy", "band_gap,
            "is_gap_direct", "m_n", "m_p".
        :param ltol: Fractional length tolerance, as defined in
            :py:class:`pymatgen.analysis.structure_matcher.StructureMatcher`.
        :param stol: Site tolerance, as the fraction of the average free length per
            atom := ( V / Nsites ) ** (1/3) as defined in
            :py:class:`pymatgen.analysis.structure_matcher.StructureMatcher`.
        :param angle_tol: Angle tolerance in degrees, as defined in
            :py:class:`pymatgen.analysis.structure_matcher.StructureMatcher`.
        :return: AlloyPair class.
        """
        # TODO: modify this method so that end user only supplies structures once

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
        ) = cls._get_oxi_state_info(anions_a, cations_a, anions_b, cations_b, alloying_element_a, alloying_element_b, )

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

        :param anions_a: Anions with oxidation state in end-point material A.
        :param cations_a: Cations with oxidation state in end-point material A.
        :param anions_b: Anions with oxidation state in end-point material B.
        :param cations_b: Cations with oxidation state in end-point material B.
        :param alloying_element_a: Element to be alloyed in end-point material A.
        :param alloying_element_b: Element to be alloyed in end-point material B.
        :return: Information about what oxidation states are present in the alloy.
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
        :param ltol: Fractional length tolerance, as defined in
            :py:class:`pymatgen.analysis.structure_matcher.StructureMatcher`.
        :param stol: Site tolerance, as the fraction of the average free length per
            atom := ( V / Nsites ) ** (1/3) as defined in
            :py:class:`pymatgen.analysis.structure_matcher.StructureMatcher`.
        :param angle_tol: Angle tolerance in degrees, as defined in
            :py:class:`pymatgen.analysis.structure_matcher.StructureMatcher`.
        :return: True or False
        """

        # TODO: allow competing_pairs kwarg which may change the behavior of this method depending on if
        # it is known that other alloy pairs exist with this composition

        if self.chemsys != structure.composition.chemical_system:
            return False

        # checking by spacegroup is useful for DISORDERED structures
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

        :param composition: Input composition of a material.
        :return: Fractional alloy content, x.
        """
        c = composition.element_composition
        if (self.alloying_element_a not in c) or (self.alloying_element_b not in c):
            raise ValueError("Provided composition does not contain required alloying elements.")
        return c[self.alloying_element_a] / (c[self.alloying_element_a] + c[self.alloying_element_b])

    def get_property_with_vegards_law(self, x: float, prop: str = "band_gap") -> Optional[float]:
        """
        Apply Vegard's law to obtain a linearly interpolated property value.

        :param x: Fractional alloy content, x.
        :param prop: Property of interest (must be defined for both end-point material A and
            end-point material B).
        :return: Interpolated property of alloy. If not defined will return None.
        """
        prop_a = getattr(self, f"{prop}_a", None) or self.properties_a.get(prop)
        prop_b = getattr(self, f"{prop}_b", None) or self.properties_b.get(prop)
        if (prop_a is None) or (prop_b is None):
            return None
        return (1 - x) * prop_a + x * prop_b

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

        :param structure_a: Pymatgen Structure for end-point material A.
        :param structure_b: Pymatgen Structure for end-point material B.
        :param ltol: Fractional length tolerance, as defined in
            :py:class:`pymatgen.analysis.structure_matcher.StructureMatcher`.
        :param stol: Site tolerance, as the fraction of the average free length per
            atom := ( V / Nsites ) ** (1/3) as defined in
            :py:class:`pymatgen.analysis.structure_matcher.StructureMatcher`.
        :param angle_tol: Angle tolerance in degrees, as defined in
            :py:class:`pymatgen.analysis.structure_matcher.StructureMatcher`.
        :return: Element to be alloyed in end-point material A and B, respectively.
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

    def as_records(self, fields: Optional[List[str]] = None) -> List[Dict]:
        """
        Convert to a record to remove _a, _b subscripts for easier plotting. Can use
        to create a pandas DataFrame.

        :param fields: Subset of attributes to return (default: None).
        :return: List of dictionaries of record information with _a and _b subscripts.
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
        Additional fields from which to build search indices for MongoDB.
        For example, it's useful to have a list of both formulas A and B
        present during a search.

        :return: A dictionary of additional fields, e.g. including
        {"formula": [self.formula_a, self.formula_b]} etc.
        """

        search = dict()

        # TODO: this is silly, note to self to make this less "clever"

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

        search["member_ids"] = [m.id_ for m in self.members]

        return search


@dataclass
class AlloySystem(MSONable):
    """
    An alloy system defined by a group of alloy pairs (defined by AlloyPair).

    Attributes:
        ids (Set[str]): A flat list of all identifiers in the
            alloy system.
        alloy_pairs (List[AlloyPair]): A list of the alloy pairs
            belonging to this system.
        alloy_id (str): A unique identifier for this alloy system.
        n_pairs (int): Number of alloys pairs in this alloy system.
        chemsys (str): The chemical system of this alloy system.
        chemsys_size (int): Number of elements in chemical system.
        pair_ids (Set[str]): A list of id pairs defining
            the given alloy pair (where ids are underscore
            delimited and given in lexicographical order of their
            reduced formula, this format matches the format
            from pair_id given in alloy pairs so is easy to
            query in a database).
        has_members (bool): Whether there are alloy members
            (defined by AlloyMember) in this alloy system.
        members (List[AlloyMember]): Flat list of alloy members
            (defined by AlloyMember) that make up the system.
        additional_members (List[AlloyMember]): Additional members of
            the system that are not a member of any one individual
            alloy pair.
    """

    ids: Set[str]
    alloy_pairs: List[AlloyPair] = field(repr=False)
    alloy_id: str = ""
    n_pairs: int = 0
    chemsys: str = ""
    chemsys_size: int = 0
    pair_ids: Set[str] = field(default_factory=set)
    has_members: bool = False
    members: List[AlloyMember] = field(default_factory=list, repr=False)
    additional_members: List[AlloyMember] = field(default_factory=list, repr=False)

    # TODO: sets of alloying_elements etc?
    # TODO: property ranges for searching?
    # property_ranges: Dict[SupportedProperties, Tuple[float, float]] = field(init=False)

    def __post_init__(self):

        # due to type changes when re-serializing from JSON via from_dict()
        if isinstance(self.ids, list):
            self.ids = set(self.ids)
        if isinstance(self.pair_ids, list):
            self.pair_ids = set(self.pair_ids)

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

        self.members = list(chain.from_iterable(pair.members for pair in self.alloy_pairs))
        self.has_members = bool(self.members)

    @staticmethod
    def systems_from_pairs(alloy_pairs: List[AlloyPair]) -> List["AlloySystem"]:
        """
        Function to construct AlloySystem class from a list of alloy pairs.

        :param alloy_pairs: A list of the alloy pairs
            belonging to this system.
        :return: A list of AlloySystem classes.
        """
        g = nx.Graph()

        for pair in alloy_pairs:
            g.add_edge(pair.id_a, pair.id_b)

        subgraphs = nx.connected_components(g)

        def get_pairs_from_id_bunch(id_bunch):
            return [pair for pair in alloy_pairs if ((pair.id_a in id_bunch) or (pair.id_b in id_bunch))]

        systems = [
            AlloySystem(ids=set(subgraph), alloy_pairs=get_pairs_from_id_bunch(subgraph)) for subgraph in subgraphs
        ]

        return systems

    def get_property(self, id_: str, prop: SupportedProperties) -> Any:
        """
        Get specified property of an alloy end-point in an alloy system.

        :param id_: Unique identifier, e.g. Materials Project material_id.
        :param prop: Specified materials property e.g. "e_above_hull",
            "formation_energy", "band_gap, "is_gap_direct", "m_n", "m_p".
        :return: Specified property for specified end-point material.
        """
        for pair in self.alloy_pairs:
            if pair.id_a == id_:
                if hasattr(pair, f"{prop}_a"):
                    return getattr(pair, f"{prop}_a")
                elif prop in pair.properties_a:
                    return pair.properties_a[prop]
                else:
                    return getattr(pair, f"{prop}")
            elif pair.id_b == id_:
                if hasattr(pair, f"{prop}_b"):
                    return getattr(pair, f"{prop}_b")
                elif prop in pair.properties_b:
                    return pair.properties_b[prop]
                else:
                    return getattr(pair, f"{prop}")

    def systems_from_filter(
            self, pair_filter: Callable[[AlloyPair], bool], origin: Optional[str] = None
    ) -> Union[List["AlloySystem"], "AlloySystem"]:
        """
        Filter the AlloySystem by a provided constraint, e.g. to remove AlloyPair
        entries that are not isoelectronic with each other. This returns a list of
        AlloySystem since, after the filter is applied, the original AlloySystem
        might be broken into several smaller AlloySystem.

        :param pair_filter: A function that takes an AlloyPair and returns a boolean,
        for example "pair_filter=lambda pair: pair.isoelectronic"
        :param origin: If supplied, will only return the system containing this identifier.
        :return: List of AlloySystem if an origin is not provided, or a single AlloySystem
        if the origin is provided
        """

        pairs = [pair for pair in self.alloy_pairs if pair_filter(pair)]

        if not origin:
            return AlloySystem.systems_from_pairs(pairs)

        g = nx.Graph()

        for pair in pairs:
            g.add_edge(pair.id_a, pair.id_b)

        subgraphs = [g.subgraph(c).copy() for c in nx.connected_components(g) if origin in c]

        def get_pairs_from_subgraph(subgraph):
            return [
                pair
                for pair in self.alloy_pairs
                if ((pair.id_a, pair.id_b) in subgraph.edges) or ((pair.id_b, pair.id_a) in subgraph.edges)
            ]

        filtered_systems = [
            AlloySystem(ids=set(subgraph), alloy_pairs=get_pairs_from_subgraph(subgraph)) for subgraph in subgraphs
        ]

        if origin:
            if len(filtered_systems) > 1:
                raise Exception(
                    "More than one filtered system contains the origin, this shouldn't happen! Debug required."
                )
            return filtered_systems[0]
        else:
            return filtered_systems

    def get_convex_hull_and_centroid(
            self, x_prop: SupportedProperties = "volume_cube_root", y_prop: SupportedProperties = "band_gap"
    ) -> Tuple[List, List, float]:
        """
        Get convex hull, centroid, and area from specified material properties.

        :param x_prop: Specified materials property (default: "volume_cube_root").
        :param y_prop: Another specified materials property (default: "band_gap").
        :return: Convex hull, centroid, and area of convex hull.
        """
        points = [(self.get_property(id_, x_prop), self.get_property(id_, y_prop)) for id_ in self.ids]

        if len(points) < 3:
            return None, None, None

        hull = MultiPoint(points).convex_hull

        return list(hull.exterior.coords), list(hull.centroid.coords)[0], hull.area

    def get_hull_trace_and_area(
            self,
            x_prop: SupportedProperties = "volume_cube_root",
            y_prop: SupportedProperties = "band_gap",
            color: Tuple[int, int, int] = (0, 0, 0),
            opacity: float = 0.2,
            colour_by_centroid: bool = False,
    ) -> Tuple[go.Trace, float]:
        """
        Get a single convex hull trace in a plotly format, and the
        area of the convex hull.

        :param x_prop: Specified materials property (default: "volume_cube_root").
        :param y_prop: Another specified materials property (default: "band_gap").
        :param color: Specified rgb color.
        :param opacity: Opacity.
        :param colour_by_centroid: Whether to color according to centroid value.
        :return: Convex hull plotly figure and the area of the hull.
        """

        hull, centroid, area = self.get_convex_hull_and_centroid(x_prop, y_prop)

        if not hull:
            return None, None

        if y_prop == "band_gap" and colour_by_centroid:
            try:
                color = ev_to_rgb(centroid[1])
            except ValueError:  # wavelength out of range
                pass

        trace = go.Scatter(
            x=[p[0] for p in hull],
            y=[p[1] for p in hull],
            fill="toself",
            fillcolor=f"rgba({color[0]},{color[1]},{color[2]},{opacity})",
            hoverinfo="skip",
            mode="none",
            showlegend=False,
        )

        return trace, area

    def plot(
            self,
            x_prop: SupportedProperties = "volume_cube_root",
            y_prop: SupportedProperties = "band_gap",
            symbol: str = "theoretical",
            column_mapping: Dict = None,
            plotly_pxline_kwargs: Dict = None,
            plotly_pxscatter_kwargs: Dict = None,
            plot_members: bool = True,
            member_plotly_pxscatter_kwargs: Dict = None,
    ) -> go.Figure:
        """
        Get plot of alloys system space for two specified material properties.

        :param x_prop: Specified materials property (default: "volume_cube_root").
        :param y_prop: Another specified materials property (default: "band_gap").
        :param symbol: Properties to designate as symbols on plot.
        :param column_mapping: Dictionary to set human-readable column names for properties.
        :param plotly_pxline_kwargs: Plotly line graph keyward arguments.
        :param plotly_pxscatter_kwargs: Plotly scatter graph keyward arguments.
        :param plot_members: Whether to plot alloy members within the alloy system.
        :param member_plotly_pxscatter_kwargs: Plotly line graph keyward arguments for members.
        :return: Plot of alloy system (AlloySystem) phase space.
        """

        data = []
        member_data = []
        column_mapping = column_mapping or {x_prop: x_prop, y_prop: y_prop, symbol: symbol}

        for pair in self.alloy_pairs:
            data.append(
                {
                    column_mapping[x_prop]: getattr(pair, f"{x_prop}_a", None) or pair.properties_a.get(x_prop),
                    column_mapping[y_prop]: getattr(pair, f"{y_prop}_a", None) or pair.properties_a.get(y_prop),
                    column_mapping[symbol]: getattr(pair, f"{symbol}_a", None) or pair.properties_a.get(symbol),
                    "id": pair.id_a,
                    "formula": pair.formula_a,
                    "pair_id": pair.pair_id,
                    "has_members": "Yes" if len(pair.members) > 0 else "No",
                }
            )
            data.append(
                {
                    column_mapping[x_prop]: getattr(pair, f"{x_prop}_b", None) or pair.properties_b.get(x_prop),
                    column_mapping[y_prop]: getattr(pair, f"{y_prop}_b", None) or pair.properties_b.get(y_prop),
                    column_mapping[symbol]: getattr(pair, f"{symbol}_b", None) or pair.properties_b.get(symbol),
                    "id": pair.id_b,
                    "formula": pair.formula_b,
                    "pair_id": pair.pair_id,
                    "has_members": "Yes" if len(pair.members) > 0 else "No",
                }
            )
            if plot_members:
                for member in pair.members:
                    member_data.append(
                        {
                            column_mapping[x_prop]: pair.get_property_with_vegards_law(x=member.x, prop=x_prop),
                            column_mapping[y_prop]: pair.get_property_with_vegards_law(x=member.x, prop=y_prop),
                            "pair_id": pair.pair_id,
                            # "formula": unicodeify(Composition.from_dict(member.composition).reduced_formula),
                            "id": member.id_,
                            "db": member.db,
                            "x": member.x,
                        }
                    )

        df = pd.DataFrame(data)

        pxline_kwargs = dict(
            line_group="pair_id",
            text="formula",
            labels="formula",
            hover_data=["formula", "id"],
            title=f"Alloy system: {self.alloy_id}",
            markers=False,
            line_dash="has_members",
            category_orders={"has_members": ["Yes", "No"]},
        )
        plotly_pxline_kwargs = plotly_pxline_kwargs or {}
        pxline_kwargs.update(plotly_pxline_kwargs)

        # initialize figure
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

        if plot_members and member_data:

            member_pxscatter_kwargs = dict(hover_data=["id", "x"])
            member_plotly_pxscatter_kwargs = member_plotly_pxscatter_kwargs or {}
            member_pxscatter_kwargs.update(member_plotly_pxscatter_kwargs)

            member_df = pd.DataFrame(member_data)

            member_fig = px.scatter(
                member_df, column_mapping[x_prop], column_mapping[y_prop], **member_pxscatter_kwargs
            )
            for trace in member_fig.data:
                fig.add_trace(trace)

        if y_prop == "band_gap":
            for ev in np.arange(1.6, 3.1, 0.05):
                fillcolor = "rgb({},{},{})".format(*ev_to_rgb((ev + ev + 0.1) / 2))
                fig.add_hrect(y0=ev, y1=ev + 0.05, fillcolor=fillcolor, layer="below", line_width=0)

        return fig

    def __len__(self):
        return len(self.pair_ids)

    def as_dict(self) -> dict:
        d = super().as_dict()
        # because JSON doesn't have a set type
        # alternative would be to use list, but set more appropriate
        d["ids"] = list(d["ids"])
        d["pair_ids"] = list(d["ids"])
        return d

    def as_dict_mongo(self):
        # do not store AlloyPairs?
        # add number of components
        # add search dict
        raise NotImplementedError


def combine_systems(alloy_systems: List[AlloySystem]) -> List[AlloySystem]:
    pass


@dataclass
class AlloySegment(MSONable):
    """
    A segment within an alloy, consisting of a segment start composition,
    segment end composition, and the corresponding pair of identifiers.

    Attributes:
        x_segment_start (float): The fractional composition of the start of a segment
            (between 0 and 1).
        x_segment_end (float): The fractional composition of the end of a segment
            (between 0 and 1).
        pair_id (str): A unique identifier for the specified alloy pair.

    """

    x_segment_start: float
    x_segment_end: float
    pair_id: str


@dataclass
class FormulaAlloyPair(MSONable):
    """
    Data class for creating a formula alloy pair which defines a set of all alloy pairs
    (of class AlloyPair) between two end-point materials A and B of the same
    formula but of different polymorphs. This class is used to find which polymorphs
    might be stable at a given alloy content x.

    Use the .from_pairs() method to automatically determine segments.

    Attributes:
        segments (List): Group of segments within an alloy, consisting of a
            segment start composition, segment end composition, and the
            corresponding pair of identifiers.
        pairs (List[AlloyPair]): List of alloy pairs that comprise a specified
            pair of formulas.

    """

    segments: List[AlloySegment]
    pairs: List[AlloyPair]

    @classmethod
    def from_pairs(cls, pairs: List[AlloyPair]) -> List["FormulaAlloyPair"]:
        """
        Gets list of FormulaAlloyPair objects for a set of AlloyPair objects

        :param pairs: List of alloy pairs that comprise a specified
            pair of formulas.
        :return: Formula alloy pairs, containing (1) input pairs and
            (2) group of segments within an alloy, consisting of a
            segment start composition, segment end composition, and the
            corresponding alloy pair identifiers.
        """

        pairs = sorted(pairs, key=lambda pair: pair.pair_formula)
        alloy_pair_groups = groupby(pairs, key=lambda pair: pair.pair_formula)

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

        pairs_ids = alloy_pair_df["pair_id"].unique()

        lines = []
        # construct so that the order of lines matches the order of pairs_ids
        for pair in pairs_ids:
            df_pair = alloy_pair_df[alloy_pair_df["pair_id"] == pair]
            a_1 = (
                df_pair[df_pair["is"] == "a"]["x"].values[0],
                df_pair[df_pair["is"] == "a"]["energy_above_hull"].values[0],
            )
            b_1 = (
                df_pair[df_pair["is"] == "b"]["x"].values[0],
                df_pair[df_pair["is"] == "b"]["energy_above_hull"].values[0],
            )
            lines.append(FormulaAlloyPair._get_line_equation(a_1, b_1))
        lines += [[-1, 0, 0], [1, 0, -1]]  # add end-points

        # define halfspaces
        halfspaces = np.array(lines)

        # define a point in alloy space with unphysically low energy
        feasible_point = np.array([0.5, -1])
        hs = HalfspaceIntersection(halfspaces, feasible_point)

        a_min = alloy_pair_df[alloy_pair_df["is"] == "a"].sort_values("energy_above_hull").iloc[0]
        a_endpoint = (a_min["x"], a_min["energy_above_hull"])
        b_min = alloy_pair_df[alloy_pair_df["is"] == "b"].sort_values("energy_above_hull").iloc[0]
        b_endpoint = (b_min["x"], b_min["energy_above_hull"])

        points = [a_endpoint, b_endpoint]
        for i in hs.intersections:
            if (i[0] > 0) & (i[0] < 1):
                points.append(i)

        hull_df = pd.DataFrame(points, columns=["x", "energy_above_hull"]).sort_values("x")
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
            pair_id = set(intersection_to_pairs[i][1]).intersection(intersection_to_pairs[i + 1][1]).pop()
            segments.append(AlloySegment(x_segment_start=x_segment_start, x_segment_end=x_segment_end, pair_id=pair_id))

        return hull_df, segments

    @staticmethod
    def _get_alloy_polymorphs_from_mp(df_alloy, old_API=False, api_key=None):

        formula_a = list(df_alloy.loc[df_alloy["is"] == "a", "formula"])[0]
        formula_b = list(df_alloy.loc[df_alloy["is"] == "b", "formula"])[0]
        fields = ["formula_pretty", "material_id", "symmetry.symbol", "energy_above_hull", "theoretical"]

        # new API
        if not old_API:
            from mp_api.client import MPRester
            with MPRester(api_key) as mpr:
                polymorphs = mpr.summary.search(
                    formula=[formula_a, formula_b],
                    fields=fields,
                )
            df_polymorphs = pd.DataFrame([{
                "formula": p.formula_pretty, "task_id": p.material_id, "spacegroup.symbol": p.symmetry.symbol, \
                "theoretical": p.theoretical, "energy_above_hull": p.energy_above_hull} \
                for p in polymorphs])
            alloy_map = {formula_a: 0, formula_b: 1}
            df_polymorphs["x"] = [alloy_map[iz] for iz in df_polymorphs["formula"]]

            # old API
        else:
            from pymatgen.ext.matproj import MPRester
            with MPRester(api_key) as mpr:
                polymorphs = mpr.query(
                    {"pretty_formula": {"$in": [formula_a, formula_b]}},
                    fields,
                )
            df_polymorphs = pd.DataFrame(polymorphs)
            df_polymorphs = df_polymorphs.rename(columns={'e_above_hull': 'energy_above_hull'})

        alloy_map = {formula_a: 0, formula_b: 1}
        df_polymorphs["x"] = [alloy_map[iz] for iz in df_polymorphs["formula"]]

        return df_polymorphs

    def plot(
            self,
            supplement_with_mp: bool = True,
            supplement_with_members: bool = True,
            add_hull_shading: bool = True,
            add_decomposition: bool = False,
            w: float = 650,
            h: float = 600,
            color: str = "pair_id",
            color_scale: str = "default",
            color_map: Dict = None,
            n_colors: int = 9,
            y_limit: float = None,
            e_above_hull_type: Literal["interpolated", "mp"] = "interpolated",
            hull_shading_opacity_factor: float = 0.2,
            api_key: str = None,
            old_API: bool = False,  # TODO: should remove old API stuff before commit
            plot_critical_lines: bool = False,
            fap_plot_domain: Tuple[float, float] = (0.22, 1),
            decomp_plot_domain: Tuple[float, float] = (0, 0.18),
    ) -> go.Figure:
        """
        Get a half-space hull plot for a specified formula alloy pair.

        :param hull_shading_opacity_factor:
        :param e_above_hull_type:
        :param decomp_plot_domain:
        :param fap_plot_domain:
        :param old_API:
        :param api_key:
        :param y_limit:
        :param color:
        :param color_map:
        :param add_decomposition:
        :param add_hull_shading:
        :param plot_critical_lines:
        :param supplement_with_mp: Whether to plot additional MP structures at end-point compositions,
            that are not defined as part of an alloy pair.
        :param supplement_with_members: Whether to plot members along alloy pair tieline.
        :param w: Plot width.
        :param h: Plot height.
        :return: Half-space hull plotly figure.
        """
        from mp_api.client import MPRester

        fields = [
            "x",
            "is",
            "formula",
            "id",
            "energy_above_hull",
            "formation_energy_per_atom",
            "spacegroup_intl_number",
            "pair_id",
            "pair_formula",
            "alloy_formula",
        ]
        df = pd.DataFrame(chain.from_iterable(pair.as_records(fields=fields) for pair in self.pairs))
        hull_df, _ = self._get_hull(df)

        # make a color map
        if not color_map:
            if color_scale == "default":
                colors = px.colors.DEFAULT_PLOTLY_COLORS
            else:
                n_colors = n_colors
                range_of_scale = (0, 1)
                colors = px.colors.sample_colorscale(
                    color_scale, n_colors,
                    low=range_of_scale[0], high=range_of_scale[1]
                )
            color_keys = df[color].unique()
            color_map = {k: colors[idx] for idx, k in enumerate(color_keys)}
            if len(color_keys) > len(colors):
                raise ValueError("The number of alloy pairs exceeds \
                the number of colors. Please supply your own color map!")

        px_fig = px.line(
            df,
            x="x",
            y="energy_above_hull",
            color=color,
            line_group="pair_id",
            line_dash_sequence=["dot"],
            color_discrete_map=color_map,
            markers=True,
            hover_data=fields,
        )

        if add_decomposition:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        else:
            fig = make_subplots(rows=1, cols=1)

        fig.add_traces(px_fig.data)
        # fig.update_layout(px_fig.layout)

        fig.add_scatter(
            x=hull_df["x"],
            y=hull_df["energy_above_hull"],
            mode="lines + markers",
            name="hull",
            opacity=0.5,
            marker_color="black",
            marker_size=10,
            hoverinfo="all",
        )
        fig.update_traces(marker=dict(size=10), line=dict(width=4))
        # TODO: just change halfspace hull
        fig.update_traces(selector={"name": "hull"}, marker=dict(size=15), line=dict(width=8))

        if supplement_with_mp:
            df_polymorphs = self._get_alloy_polymorphs_from_mp(df, api_key=api_key, old_API=old_API)
            fig.add_scatter(
                x=df_polymorphs["x"],
                y=df_polymorphs["energy_above_hull"],
                mode="markers",
                marker_color="black",
                marker_size=5,
                name="MP polymorphs",
                hoverinfo="text",
                hovertext=list(np.array(df_polymorphs))
            )
            # TODO: figure out how to reorder so that there's no bug!
            # fig.data = tuple(list(fig.data[1:]) + [fig.data[0]])

        if supplement_with_members:

            mp_member_ids = set()
            for pair in self.pairs:
                for member in pair.members:
                    if member.db == "mp":
                        mp_member_ids.add(member.id_)
                # mp_member_ids.add({member.id_ for member in pair.members if member.db == "mp"})
            member_records = []
            if mp_member_ids:
                with MPRester(api_key) as mpr:
                    e_above_hulls = {doc.material_id: doc.energy_above_hull for doc in
                                     mpr.thermo.search_thermo_docs(material_ids=list(mp_member_ids))}

                for pair in self.pairs:
                    for i, member in enumerate(pair.members):
                        if member.db == "mp":
                            member_records.append({
                                "member_formula": Composition(member.composition).reduced_formula,
                                "pair_id": pair.pair_id,  # why
                                "pair_formula": pair.pair_formula,  # why
                                "alloy_formula": pair.alloy_formula,  # why
                                "spacegroup_intl_number": pair.spacegroup_intl_number_a,  # why
                                "x": member.x,
                                "interpolated_energy_above_hull": pair.get_property_with_vegards_law(member.x,
                                                                                                     "energy_above_hull"),
                                "mp_energy_above_hull": e_above_hulls.get(member.id_, "Unknown")
                            })

            if member_records:
                member_df = pd.DataFrame(member_records)
                # if e_above_hull_type == "interpolated":
                #     e_above_hull = member_df["interpolated_energy_above_hull"]
                # elif e_above_hull_type == "mp":
                #     e_above_hull = member_df["mp_energy_above_hull"]
                # else:
                #     raise ValueError("Must specify ehull type!")

                if color == "spacegroup_intl_number":
                    member_color = "spacegroup_intl_number_a"

                color_keys = member_df[color].unique()
                for color_key in color_keys:
                    member_df_plot = member_df[member_df[color] == color_key]
                    #     member_df["color_rgb"][i] = color_map[color_key]
                    e_above_hull = member_df_plot[e_above_hull_type + "_energy_above_hull"]
                    pair_id = member_df_plot["pair_id"].unique()[0]
                    fig.add_scatter(
                        x=member_df_plot["x"], y=e_above_hull, mode="markers",
                        marker={"color": color_map[color_key], "symbol": "square"},
                        name=f"member of {pair_id}",
                    )

        fig.update_layout(title=df["alloy_formula"][0], legend_title_text="")
        fig.update_yaxes(title="<i>E</i><sub>hull</sub> (eV/atom)")
        if not y_limit:
            y_limit = df["energy_above_hull"].max()
        fig.update_yaxes(range=(-y_limit * 0.15, y_limit + (y_limit * 0.15)))
        fig.update_layout(
            autosize=False,
            xaxis=dict(showgrid=False, mirror=True),
            yaxis=dict(showgrid=False, mirror=True),
            height=h,
            width=w,
            hovermode="closest",
            template="simple_white",
            font=dict(family="Helvetica", size=16, color="black"),
        )

        if add_hull_shading:
            fig = self._add_halfspace_hull_shading(
                fig,
                opacity_factor=hull_shading_opacity_factor,
                reorder_traces=False
            )

        if add_decomposition:
            fig.update_traces(row=1, col=1)
            decomp_df = self._get_decomp_df(api_key=api_key)
            fig = self._add_decomp_figure(
                fig,
                decomp_df,
                plot_critical_lines=plot_critical_lines,
                fap_plot_domain=fap_plot_domain,
                decomp_plot_domain=decomp_plot_domain,
                row_decomp=2,
                h=h,
                w=w
            )

        return fig

    def _get_decomp_df(
            self,
            api_key: str = None,

    ) -> Tuple[pd.DataFrame, List]:
        """
        Creates a dataframe of thermodynamic decomposition products for a given FormulaAlloyPair
        for plotting purposes. Data comes from Materials Project.

        :return:
        """
        from mp_api.client import MPRester

        # one example pair -- note this is a AlloyPair method, should be under different class?
        test_pair = self.pairs[0]

        formulas = [test_pair.formula_a, test_pair.formula_b]
        elems = list(set([elem for elem in test_pair.chemsys.split("-")]))
        with MPRester(api_key) as mpr:
            mp_entries = mpr.get_entries_in_chemsys(elems)
        phase_diagram = PhaseDiagram(mp_entries)

        step_size = 0.01
        all_decomp_products = set()
        records = []
        decomp_records = []

        for x in np.arange(0, 1, step_size):
            comp = (1 - float(x)) * Composition(formulas[0]) + float(x) * Composition(formulas[1])
            decomp = phase_diagram.get_decomposition(comp)
            all_decomp_products |= {p.composition.reduced_formula for p in decomp}
            records.append(decomp)
            for entry in decomp:
                decomp_records.append({
                    "x": x,
                    "entry_id": entry.entry_id,
                    "formula": entry.name,
                    "fraction": decomp[entry],
                    "critical": False,
                })
        # critical_x = []
        for comp in phase_diagram.get_critical_compositions(Composition(formulas[0]), Composition(formulas[1])):
            if comp == Composition(formulas[0]):
                x = 0
            elif comp == Composition(formulas[1]):
                x = 1
            else:
                test_pair.get_x(comp)
            decomp = phase_diagram.get_decomposition(comp)
            all_decomp_products |= {p.composition.reduced_formula for p in decomp}
            records.append(decomp)
            # critical_x.append(x)
            for entry in decomp:
                decomp_records.append({
                    "x": x,
                    "entry_id": entry.entry_id,
                    "formula": entry.name,
                    "fraction": decomp[entry],
                    "critical": True,
                })
        df = pd.DataFrame(decomp_records)

        return df

    @staticmethod
    def _add_decomp_figure(
            fig,
            df,
            plot_critical_lines=False,
            fap_plot_domain=(0.22, 1),
            decomp_plot_domain=(0, 0.18),
            h=600,
            w=650,
            row_decomp=2,
    ) -> go.Scatter():
        """
        Add decomposition information to a FormulaAlloyPair plot

        :param row_decomp:
        :param decomp_plot_domain:
        :param fap_plot_domain:
        :param plot_critical_lines:
        :type fig: go.Scatter()
        """
        color_map = _get_colormap_from_keys(df["formula"].unique())
        all_decomp_products = df.sort_values(by="x").formula.unique()
        for i, formula in enumerate(all_decomp_products):
            entry_id = df["entry_id"][i]
            df_f = df[df["formula"] == formula]
            #     fig.add_trace(go.Scatter(x=df_f["x"], y=df_f["fraction"], fill='tonexty')) # fill down to xaxis
            fig.add_trace(go.Scatter(
                x=df_f["x"], y=df_f["fraction"],
                hoverinfo='x+y',
                mode='lines',
                name=f"{formula}: {entry_id}",
                line=dict(width=0, color=color_map[formula]),
                stackgroup='one',  # define stack group,

            ), row=row_decomp, col=1)

        if plot_critical_lines:
            for x in set(df[df["critical"]]["x"]):
                fig.add_trace(go.Scatter(
                    x=(x, x), y=(0, 1),
                    mode='lines',
                    name="critical point",
                    line=dict(width=1, color="black", dash='dot'),
                    showlegend=False,
                ), row=row_decomp, col=1)
        fig.update_yaxes(range=(0, 1), title="fraction", row=row_decomp, col=1, mirror=True)
        fig.update_xaxes(title="<i>x</i>", row=row_decomp, col=1, mirror=True)
        fig.update_layout(
            font=dict(family="Helvetica", size=14, color="black"),
            yaxis2=dict(domain=decomp_plot_domain), yaxis1=dict(domain=fap_plot_domain),
            height=h, width=w,
        )

        return fig

    @staticmethod
    def _add_halfspace_hull_shading(
            fig,
            e_hull_limit=0.1,
            hull_tuple="auto",
            shades=50,
            rgb=[100, 100, 100],
            reorder_traces=True,
            opacity_factor=0.2,
            opacity_exp=1 / 2
    ):
        """
        Method to add a shaded halfspace hull window to FormulaAlloyPair plot

        :param fig:
        :param hull_tuple:
        :param shades:
        :param rgb:
        :param reorder_traces:
        :param e_hull_limit:
        :param opacity_factor:
        :param opacity_exp:
        :return: figure
        """

        if not fig:
            fig = go.Figure()
        if hull_tuple == "auto":
            x_hull, y_hull = FormulaAlloyPair.get_critical_points_data(fig)

            # for data in fig.data:
            #     data = data.to_plotly_json()
            #     if data["name"] == "hull":
            #         critical_points = data
            # x_hull, y_hull = critical_points["x"], critical_points["y"]

        for n in range(shades - 1):
            i = n + 1
            opac = (1 - (i / shades) ** opacity_exp) * opacity_factor
            fig.add_trace(go.Scatter(
                x=list(x_hull) + list(x_hull)[::-1],
                y=list(y_hull + e_hull_limit * (i - 1) / (shades - 1)) + list(
                    y_hull + e_hull_limit * (i) / (shades - 1))[
                                                                         ::-1],
                fill="tonexty",
                fillcolor='rgba({}, {}, {}, {})'.format((rgb[0]), (rgb[1]), (rgb[2]), (opac)),
                line=dict(color='rgba({}, {}, {}, {})'.format((rgb[0]), (rgb[1]), (rgb[2]), (opac)),
                          width=0, ),
                showlegend=False,
                name="hull_shading",
                mode="lines",
            ),
                row=1,
                col=1
            )

        if reorder_traces:
            fig.data = tuple([fig.data[-1]] + list(fig.data[:-1]))

        return fig

    # may not be necessary?
    @staticmethod
    def get_critical_points_data(fig):
        """
        Function to get the critical points of the halfspace hull for a FormulaAlloyPair figure
        :param fig:
        :return:
        """

        for data in fig.data:
            data = data.to_plotly_json()
            if data["name"] == "hull":
                critical_points = data
        return critical_points["x"], critical_points["y"]

    def __str__(self):
        return f"FormulaAlloyPair {self.pairs[0].alloy_formula}"


def _get_colormap_from_keys(
        color_keys,
        color_scale="greys",
        range_of_scale=(0.1, 0.9)
):
    """
    A general function to write a plotly colormap from a set of keys

    :param color_keys:
    :param color_scale:
    :param range_of_scale:
    :return:
    """
    n_colors = len(color_keys)

    colors = px.colors.sample_colorscale(
        color_scale, n_colors,
        low=range_of_scale[0], high=range_of_scale[1]
    )
    color_map = {k: colors[idx] for idx, k in enumerate(color_keys)}

    return color_map
