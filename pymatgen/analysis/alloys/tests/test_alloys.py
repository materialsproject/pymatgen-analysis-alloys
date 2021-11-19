import unittest

from pymatgen.analysis.alloys.mymodule import myfunc


class FuncTest(unittest.TestCase):
    def test_myfunc(self):
        self.assertEqual(myfunc(1, 1), 2)
