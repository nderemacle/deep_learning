import unittest

import tensorflow as tf

import core.deep_learning.env as env
from core.deep_learning.base_operator import BaseOperator


class Operator(BaseOperator):
    def __init__(self, name: str):
        super().__init__(name)
        self.has_build = False
        self.has_restore = False
        self.x: tf.placeholder = None

    def build(self):
        super().build()

    def _build(self):
        self.has_build = True
        self.x = tf.placeholder(tf.float32, name="x")

    def restore(self):
        self.has_restore = True


class TestAbstractOperator(tf.test.TestCase):

    def testBuild(self):
        op = Operator(name="MyOp")
        env.RESTORE = False
        op.build()

        self.assertTrue(op.has_build)
        self.assertFalse(op.has_restore)
        self.assertEqual(op.x.name, "MyOp/x:0")

    def testRestore(self):
        op = Operator(name="MyOp")
        env.RESTORE = True
        op.build()

        self.assertFalse(op.has_build)
        self.assertTrue(op.has_restore)


if __name__ == '__main__':
    unittest.main()
