from pathlib import Path
from pymatgen.core import Structure

from pymatgen.analysis.alloys.core import AlloyPair
from monty.serialization import loadfn


mp_661: Structure = loadfn(Path(__file__).parent / "mp-661.json")
mp_804: Structure = loadfn(Path(__file__).parent / "mp-804.json")


def test_successful_alloy_pair_construction():

    mp_661_without_oxi_state = mp_661.copy()
    mp_661_without_oxi_state.remove_oxidation_states()

    mp_804_without_oxi_state = mp_804.copy()
    mp_804_without_oxi_state.remove_oxidation_states()

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

    mp_661_without_oxi_state = mp_661.copy()
    mp_661_without_oxi_state.remove_oxidation_states()

    mp_804_without_oxi_state = mp_804.copy()
    mp_804_without_oxi_state.remove_oxidation_states()

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

    mp_661_without_oxi_state = mp_661.copy()
    mp_661_without_oxi_state.remove_oxidation_states()

    mp_804_without_oxi_state = mp_804.copy()
    mp_804_without_oxi_state.remove_oxidation_states()

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