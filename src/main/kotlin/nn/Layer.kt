package nn

import engine.Value

class Layer(private val nin: Int, private val nout: Int, private val nonlin: Boolean = true) : Module() {
    private val neurons: List<Neuron> = List(nout) { Neuron(nin, nonlin) }

    override fun parameters(): List<Value> {
        return neurons.flatMap { it.parameters() }
    }

    operator fun invoke(x: List<Value>): List<Value> {
        val out = neurons.map { it(x) }
        return if (out.size == 1) listOf(out[0]) else out
    }

    override fun toString(): String {
        return "nn.Layer of [${neurons.joinToString()}]"
    }
}