package engine.extensions

import engine.Value

// Extension functions to accomplish the same as python's e.g. __radd__
operator fun Double.plus(other: Value): Value {
    return other + this
}

operator fun Double.times(other: Value): Value {
    return other * this
}

operator fun Double.div(other: Value): Value {
    return other.pow(-1.0) * this
}