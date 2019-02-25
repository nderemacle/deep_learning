import tensorflow as tf

import core.deep_learning.env as env
from core.deep_learning.abstract_operator import AbstractOperator


class Operator(AbstractOperator):
    def __init__(self, name: str):
        super().__init__(name)
        self.has_build = False
        self.has_restore = False

    def build(self):
        super().build()

    def _build(self):
        self.has_build = True

    def restore(self):
        self.has_restore = True


class TestAbstractOperator(tf.test.TestCase):

    def testBuild(self):
        op = Operator(name="MyOp")
        env.RESTORE = False
        op.build()

        self.assertTrue(op.has_build)
        self.assertFalse(op.has_restore)

    def testRestore(self):
        op = Operator(name="MyOp")
        env.RESTORE = True
        op.build()

        self.assertFalse(op.has_build)
        self.assertTrue(op.has_restore)
