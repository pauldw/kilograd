// Based on https://github.com/karpathy/micrograd

import engine.Value
import engine.extensions.*

import nn.MLP
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

fun graphDemo() {
    var a = Value(-4.0)
    var b = Value(2.0)
    var c = a + b
    var d = a * b + b.pow(3.0)
    c += c + 1.0
    c += 1.0 + c + (-a)
    d += d * 2.0 + (b + a).relu()
    d += 3.0 * d + (b - a).relu()
    var e = c - d
    var f = e.pow(2.0)
    var g = f / 2.0
    g += 10.0 / f

    println("${g.data}") // prints 24.7041, the outcome of this forward pass
    g.backward()
    println("${a.grad}") // prints 138.8338, i.e. the numerical value of dg/da
    println("${b.grad}") // prints 645.5773, i.e. the numerical value of dg/db
}

fun makeMoons(n: Int, noise: Double = 0.1): Pair<List<List<Double>>, List<Int>> {
    // Based on https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html source
    val range = 0..n
    val outerx = range.map{cos(PI * it/range.last)}
    val outery = range.map{sin(PI * it/range.last)}
    val innerx = range.map{1.0 - cos(PI * it/range.last)}
    val innery = range.map{1.0 - sin(PI * it/range.last) - 0.5}

    val X = (outerx + innerx).zip(outery + innery).map{ listOf(it.first, it.second) }.toTypedArray()
    val y = (range.map{-1} + range.map{1}).toIntArray()

    // TODO add noise

    return Pair(X.toList(), y.toList())
}

fun loss(model: MLP, X: List<List<Double>>, y: List<Int>) : Pair<Value, Double> {
    val scores = X.map{ model(it.map{itt -> Value(itt)}) } // TODO how do we feel about "itt"?

    // svm max-margin loss
    val losses = scores.zip(y).map{ (score, label) -> (1.0 + -label.toDouble()*score[0]).relu() }
    val dataLoss = losses.reduce{ acc, loss -> acc + loss } / losses.size.toDouble() // TODO add .sum() extension for List<Values>
    val regLoss = 1e-4 * model.parameters().map{ it.pow(2.0) }.reduce{ acc, loss -> acc + loss } / model.parameters().size.toDouble()
    val totalLoss = dataLoss + regLoss

    val accuracy = scores.zip(y)
        .map { (score, label) ->
            if (score[0].data > 0.0 && label == 1) 1
            else if (score[0].data <= 0.0 && label == -1) 1
            else 0 }
        .sum().toDouble() / y.size.toDouble()

    return Pair(totalLoss, accuracy)
}

fun moonDemo() {
    val (X, y) = makeMoons(50)
    var model = MLP(2, listOf(16, 16, 1))

    println("${model.parameters().size} parameters.")

    val (totalLoss, acc) = loss(model, X, y)
    println("initial loss $totalLoss, accuracy $acc")

    (0..100).forEach {k ->
        // Forward
        val (totalLoss, accuracy) = loss(model, X, y)

        // Backward
        model.zeroGrad()
        totalLoss.backward()

        // Update (SGD)
        val learningRate = 1.0 - 0.9*k/100
        model.parameters().forEach{p ->
            p.data -= (learningRate * p.grad)
        }

        println("step $k: loss ${totalLoss.data}, accuracy $accuracy")
    }
}

fun main(args: Array<String>) {
    println("Graph demo: ")
    graphDemo()
    println("----")

    println("Moon demo: ")
    moonDemo()
    println("----")
}