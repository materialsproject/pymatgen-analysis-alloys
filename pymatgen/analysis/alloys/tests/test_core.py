from pathlib import Path
from typing import Dict

from pytest import raises
from monty.serialization import loadfn
from pymatgen.core import Structure, Composition

from pymatgen.analysis.alloys.core import AlloyPair


mp_661: Structure = loadfn(Path(__file__).parent / "AlN_mp-661.json")
mp_661_without_oxi_state = mp_661.copy()
mp_661_without_oxi_state.remove_oxidation_states()

mp_804: Structure = loadfn(Path(__file__).parent / "GaN_mp-804.json")
mp_804_without_oxi_state = mp_804.copy()
mp_804_without_oxi_state.remove_oxidation_states()

mp_1700: Structure = loadfn(Path(__file__).parent / "AlN_mp-1700.json")
mp_1700_without_oxi_state = mp_1700.copy()
mp_1700_without_oxi_state.remove_oxidation_states()

mp_830: Structure = loadfn(Path(__file__).parent / "GaN_mp-830.json")
mp_830_without_oxi_state = mp_830.copy()
mp_830_without_oxi_state.remove_oxidation_states()

al_ga_n: Dict[str, Structure] = loadfn(Path(__file__).parent / "Al-Ga-N.json")


def test_successful_alloy_pair_construction():

    pair = AlloyPair.from_structures(
        (mp_661_without_oxi_state, mp_804_without_oxi_state),
        (mp_661, mp_804),
        ("mp-661", "mp-804"),
        properties=({}, {})
    )

    assert pair.pair_formula == "AlN_GaN"
    assert pair.pair_id == "mp-661_mp-804"

    assert pair.alloying_species_a == "Al3+"
    assert pair.alloying_species_b == "Ga3+"

    assert pair.cations_a == ["Al3+"]
    assert pair.cations_b == ["Ga3+"]
    assert pair.anions_a == ["N3-"]
    assert pair.anions_b == ["N3-"]

    assert pair.isoelectronic is True

    assert pair.observer_elements == ["N"]
    assert pair.observer_species == ["N3-"]


def test_successful_alloy_pair_construction_without_oxidation_states():

    pair = AlloyPair.from_structures(
        (mp_661_without_oxi_state, mp_804_without_oxi_state),
        (mp_661_without_oxi_state, mp_804_without_oxi_state),
        ("mp-661", "mp-804"),
        properties=({}, {})
    )

    assert pair.pair_formula == "AlN_GaN"
    assert pair.pair_id == "mp-661_mp-804"

    assert pair.alloying_species_a is None
    assert pair.alloying_species_b is None

    assert pair.isoelectronic is None

    assert pair.observer_elements == ["N"]
    assert pair.observer_species == []


def test_successful_alloy_pair_construction_with_mixed_oxidation_states():

    pair = AlloyPair.from_structures(
        (mp_661_without_oxi_state, mp_804_without_oxi_state),
        (mp_661, mp_804_without_oxi_state),
        ("mp-661", "mp-804"),
        properties=({}, {})
    )

    assert pair.pair_formula == "AlN_GaN"
    assert pair.pair_id == "mp-661_mp-804"

    assert pair.alloying_species_a is None
    assert pair.alloying_species_b is None

    assert pair.isoelectronic is None

    assert pair.observer_elements == ["N"]
    assert pair.observer_species == []


def test_membership():

    wz_pair = AlloyPair.from_structures(
        (mp_661_without_oxi_state, mp_804_without_oxi_state),
        (mp_661, mp_804),
        ("mp-661", "mp-804"),
        properties=({}, {})
    )

    zb_pair = AlloyPair.from_structures(
        (mp_1700_without_oxi_state, mp_830_without_oxi_state),
        (mp_1700, mp_830),
        ("mp-1700", "mp-830"),
        properties=({}, {})
    )

    wz_human_checked_members = {
        "mp-1228943": True,  # wurtzite
        "mp-1019508": False,  # zincblende
        "mp-1228436": True,  # wurtzite
        "mp-1228894": True,  # wurtzite
        "mp-1008556": False,  # zincblende
        "mp-1019378": False,  # zincblende
        "mp-1228953": True,  # wurtzite
    }

    wz_computed_membership = {}
    for mpid, structure in al_ga_n.items():
        wz_computed_membership[mpid] = wz_pair.is_member(structure)

    assert wz_computed_membership == wz_human_checked_members

    zb_human_checked_members = {
        "mp-1228943": False,  # wurtzite
        "mp-1019508": True,  # zincblende
        "mp-1228436": False,  # wurtzite
        "mp-1228894": False,  # wurtzite
        "mp-1008556": True,  # zincblende
        "mp-1019378": True,  # zincblende
        "mp-1228953": False,  # wurtzite
    }

    zb_computed_membership = {}
    for mpid, structure in al_ga_n.items():
        zb_computed_membership[mpid] = zb_pair.is_member(structure)

    assert zb_computed_membership == zb_human_checked_members


def test_get_x():

    pair = AlloyPair.from_structures(
        (mp_661_without_oxi_state, mp_804_without_oxi_state),
        (mp_661, mp_804),
        ("mp-661", "mp-804"),
        properties=({}, {})
    )

    c1 = Composition("AlGaN2")
    c2 = Composition({'Al3+': 1, 'Ga3+': 1, 'N3-': 2})
    c3 = Composition("Al2GaN3")
    c4 = Composition("Al2GaN2")  # incompatible, i.e. off-stoichiometric
    c5 = Composition("AlGa0.5N1.5")   # incompatible, i.e. off-stoichiometric

    assert pair.get_x(c1) == 0.5
    assert pair.get_x(c2) == 0.5
    assert pair.get_x(c3) == 2/3
    with raises(ValueError):
        pair.get_x(c4)
    with raises(ValueError):
        pair.get_x(c5)
