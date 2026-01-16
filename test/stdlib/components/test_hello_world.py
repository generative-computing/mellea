from mellea.stdlib.components import Instruction


def test_empty_instr():
    i = Instruction()
    print(i)


def test_instr_template():
    i = Instruction(description="What is 1+1?")
    print(i)
