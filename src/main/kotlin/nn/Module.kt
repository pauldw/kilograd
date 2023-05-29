package nn

import engine.Value

abstract class Module {
    open fun zeroGrad() {
        parameters().forEach { p -> p.grad = 0.0 }
    }

    open fun parameters(): List<Value> {
        return emptyList()
    }
}


