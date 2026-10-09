// Prefill chunk contract on Android: LlmModule.generate() with prompts of exactly n tokens (prompt files made on the
// host with the same tokenizer, no BOS), then the longest prompt sent as prefillPrompt() pieces of at most `bound`
// tokens followed by generate(""). Usage: PrefillAB <model.pte> <tokenizer> <prompt dir> <bound> <n>...
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmGenerationConfig
import org.pytorch.executorch.extension.llm.LlmModule
import java.io.File
import kotlin.system.exitProcess

fun run(block: (LlmCallback) -> Unit): Pair<String, String?> {
    val sb = StringBuilder()
    val err = try {
        block(object : LlmCallback { override fun onResult(result: String) { sb.append(result) } }); null
    } catch (t: Throwable) { "${t.javaClass.simpleName}: ${t.message}" }
    return sb.toString() to err
}

fun main(args: Array<String>) {
    val (model, tok, dir) = Triple(args[0], args[1], args[2])
    val bound = args[3].toInt()
    val sizes = args.drop(4).map { it.toInt() }
    val module = LlmModule(model, tok, 0.0f)
    module.load()
    fun cfg(@Suppress("UNUSED_PARAMETER") n: Int) = LlmGenerationConfig.create().seqLen(4).echo(false).temperature(0.0f).build()
    val out = StringBuilder()
    for (n in sizes) {
        module.resetContext()
        val prompt = File("$dir/p$n.txt").readText()
        val (text, err) = run { cb -> module.generate(prompt, cfg(n), cb) }
        out.append("RESULT generate n=$n ${if (err == null && text.isNotEmpty()) "ok" else "FAIL"} | out=${text.take(40).replace("\n", "\\n")} | err=${err?.take(160)}\n")
    }
    val n = sizes.max()
    module.resetContext()
    val pieces = mutableListOf<Int>(); var left = n
    while (left > 0) { pieces.add(minOf(bound, left)); left -= pieces.last() }
    // Every piece but the last through prefillPrompt(), the last through generate(), so no call sees more than
    // `bound` tokens. The 1.5.1 JNI passes only seqLen (maxNewTokens is dropped), and with get_max_seq_len <
    // get_max_context_len the runner resolves the budget from position 0 (text_llm_runner.cpp, effective_pos), so
    // seqLen(4) means 4 new tokens wherever the prompt ends.
    val (text, err) = run { cb ->
        for (p in pieces.dropLast(1)) module.prefillPrompt(File("$dir/p$p.txt").readText())
        module.generate(File("$dir/p${pieces.last()}.txt").readText(), cfg(n), cb)
    }
    out.append("RESULT workaround n=$n pieces=$pieces ${if (err == null && text.isNotEmpty()) "ok" else "FAIL"} | out=${text.take(40).replace("\n", "\\n")} | err=$err\n")
    System.out.println(); print(out); System.out.flush(); exitProcess(0)
}
