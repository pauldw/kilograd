package nn

import engine.Value

class MLP(nin: Int, nouts: List<Int>) : Module() {
    private val layers: List<Layer> = List(nouts.size) {
        val nonlin = it != nouts.size - 1
        Layer(if (it == 0) nin else nouts[it - 1], nouts[it], nonlin)
    }

    override fun parameters(): List<Value> {
        return layers.flatMap { it.parameters() }
    }

    operator fun invoke(x: List<Value>): List<Value> {
        var output = x
        for (layer in layers) {
            output = layer(output)
        }
        return output
    }

    override fun toString(): String {
        return "nn.MLP of [${layers.joinToString()}]"
    }
}