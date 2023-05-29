package engine// Based on https://github.com/karpathy/micrograd/commit/5bb639209a5217b543d899dfc23ea968252fa9c1

class Value(var data: Double, private val children: Set<Value> = emptySet(), private val op: String = "") {
    var grad: Double = 0.0
    var backwardOp: () -> Unit = {}
    private val prev: Set<Value> = HashSet(children)

    operator fun plus(other: Value): Value {
        val outData = data + other.data
        val out = Value(outData, setOf(this, other), "+")

        out.backwardOp = {
            this.grad += out.grad
            other.grad += out.grad
        }

        return out
    }

    operator fun plus(other: Double): Value {
        return this + Value(other)
    }

    operator fun times(other: Value): Value {
        val outData = data * other.data
        val out = Value(outData, setOf(this, other), "*")

        out.backwardOp = {
            this.grad += other.data * out.grad
            other.grad += this.data * out.grad
        }

        return out
    }

    operator fun unaryMinus(): Value {
        return this * -1.0
    }

    operator fun minus(other: Value): Value {
        return this + (-other)
    }

    // Not sure if this will work, changed from source
    operator fun times(other: Double): Value {
        return this * Value(other)
    }

    operator fun div(other: Double): Value {
        val outData = data / other
        val out = Value(outData, setOf(this), "/")

        out.backwardOp = {
            this.grad += out.grad / other
        }

        return out
    }

    fun pow(other: Double): Value {
        val outData = Math.pow(data, other)
        val out = Value(outData, setOf(this), "**$other")

        out.backwardOp = {
            this.grad += other * Math.pow(this.data, other - 1) * out.grad
        }

        return out
    }

    fun relu(): Value {
        val out = Value(if (data < 0) 0.0 else data, setOf(this), "ReLU")

        out.backwardOp = {
            this.grad += if (out.data > 0) out.grad else 0.0
        }

        return out
    }

    fun backward() {
        val topo = mutableListOf<Value>()
        val visited = mutableSetOf<Value>()
        fun buildTopo(v: Value) {
            if (!visited.contains(v)) {
                visited.add(v)
                v.prev.forEach { child -> buildTopo(child) }
                topo.add(v)
            }
        }
        buildTopo(this)

        grad = 1.0
        topo.reversed().forEach { v ->
            v.backwardOp()
            val j=0
        }
    }

    override fun toString(): String {
        return "Value(data=${data}, grad=${grad})"
    }
}

