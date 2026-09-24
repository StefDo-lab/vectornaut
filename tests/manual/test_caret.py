import sympy as sp
from sympy.parsing.sympy_parser import standard_transformations, convert_xor

def test_caret_parsing():
    # 1. Test standard equation parsing with caret
    local_ns = {'x': sp.Symbol('x'), 'y': sp.Symbol('y')}
    expr_str = "x^2 + y^2"
    
    # Before the fix, this would parse ^ as XOR, which in sympy is: sp.Xor(x, 2) + sp.Xor(y, 2)
    # With transformations, it should parse ^ as power: x**2 + y**2
    expr = sp.parse_expr(expr_str, local_dict=local_ns, transformations=(standard_transformations + (convert_xor,)))
    print(f"Parsed '{expr_str}' -> {expr}")
    assert expr == sp.Symbol('x')**2 + sp.Symbol('y')**2, "Failed to parse caret as exponentiation"
    print("Success: Caret parsing works correctly!")

if __name__ == "__main__":
    test_caret_parsing()
