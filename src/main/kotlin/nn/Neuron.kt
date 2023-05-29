package nn

import engine.Value
import kotlin.random.Random

class Neuron(private val nin: Int, private val nonlin: Boolean = true) : Module() {
    private val w: List<Value> = List(nin) { Value(Random.nextDouble(-1.0, 1.0)) }
    private val b: Value = Value(0.0)

    override fun parameters(): List<Value> {
        return w + b
    }

    operator fun invoke(x: List<Value>): Value {
        val act = w.zip(x).fold(b) { acc, (wi, xi) -> acc + wi * xi }
        return if (nonlin) act.relu() else act
    }

    override fun toString(): String {
        return "${if (nonlin) "ReLU" else "Linear"}nn.Neuron(${w.size}) of [${w.joinToString()}]"
    }
}