import { useEffect, useRef, useState, type FormEvent } from 'react'
import './App.css'

type Message = {
  id: number
  role: 'user' | 'assistant'
  text: string
  error?: boolean
}

type CompleteResponse = { sentence: string } | { error: string }

type Sampling = { temperature: number; top_k: number; top_p: number }

const DEFAULT_SAMPLING: Sampling = { temperature: 1.0, top_k: 0, top_p: 0.9 }

type ModelInfo = {
  config: { vocab_size: number; d_model: number; context_len: number; n_heads: number; residual: boolean }
  layers: { name: string; shape: string; formula: string; calculation: string; params: number }[]
  total: number
  equation: string
  sampling?: Sampling
}

const fmt = (n: number) => n.toLocaleString('en-US')

function ParametersPanel({ info }: { info: ModelInfo }) {
  const { vocab_size, d_model, context_len, n_heads, residual } = info.config
  return (
    <section className="params" aria-label="Parameter calculation">
      <p className="params-config">
        vocab_size = {vocab_size} · d_model = {d_model} · context_len = {context_len}
        {n_heads > 0 && ` · n_heads = ${n_heads} (head size ${d_model / n_heads})`}
        {residual && ' · residual: x + attention(x)'}
      </p>
      <table>
        <thead>
          <tr>
            <th>Layer</th>
            <th>Formula</th>
            <th>Calculation</th>
            <th className="num">Parameters</th>
          </tr>
        </thead>
        <tbody>
          {info.layers.map((l) => (
            <tr key={l.name}>
              <td>{l.name}</td>
              <td className="muted">{l.formula}</td>
              <td>{l.calculation}</td>
              <td className="num">{fmt(l.params)}</td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr>
            <td colSpan={3}>Total</td>
            <td className="num">{fmt(info.total)}</td>
          </tr>
        </tfoot>
      </table>
      <p className="params-equation">{info.equation}</p>
    </section>
  )
}

function SamplingControls({ value, onChange }: { value: Sampling; onChange: (s: Sampling) => void }) {
  const greedy = value.temperature === 0
  return (
    <section className="sampling" aria-label="Sampling settings">
      <label title="Divides the scores before softmax. 0 = greedy (always the most likely word), below 1 = safer, above 1 = more random.">
        <span>Temperature</span>
        <input
          type="range" min={0} max={2} step={0.1}
          value={value.temperature}
          onChange={(e) => onChange({ ...value, temperature: Number(e.target.value) })}
        />
        <output>{greedy ? '0 · greedy' : value.temperature.toFixed(1)}</output>
      </label>
      <label title="Keep only the k most likely words. 0 = off." className={greedy ? 'off' : ''}>
        <span>Top-k</span>
        <input
          type="range" min={0} max={20} step={1}
          value={value.top_k}
          disabled={greedy}
          onChange={(e) => onChange({ ...value, top_k: Number(e.target.value) })}
        />
        <output>{value.top_k === 0 ? 'off' : value.top_k}</output>
      </label>
      <label title="Keep the most likely words until their probabilities add up to p. 1.0 = off." className={greedy ? 'off' : ''}>
        <span>Top-p</span>
        <input
          type="range" min={0.05} max={1} step={0.05}
          value={value.top_p}
          disabled={greedy}
          onChange={(e) => onChange({ ...value, top_p: Number(e.target.value) })}
        />
        <output>{value.top_p >= 1 ? 'off' : value.top_p.toFixed(2)}</output>
      </label>
    </section>
  )
}

// Only the latest message (plus the sampling settings) goes to the backend; the history stays in the browser.
async function askModel(message: string, sampling: Sampling): Promise<{ text: string; error?: boolean }> {
  try {
    const res = await fetch('/api/complete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: message, ...sampling }),
    })
    const data: CompleteResponse = await res.json()
    if ('error' in data) return { text: data.error, error: true }
    if (!res.ok) return { text: `Request failed (${res.status}).`, error: true }
    return { text: data.sentence }
  } catch {
    return { text: 'Could not reach the server. Is `python server.py` running?', error: true }
  }
}

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [info, setInfo] = useState<ModelInfo | null>(null)
  const [showParams, setShowParams] = useState(true)
  const [sampling, setSampling] = useState<Sampling>(DEFAULT_SAMPLING)
  const nextId = useRef(0)

  // Model size + parameter calculation, fetched once when the page opens.
  useEffect(() => {
    fetch('/api/info')
      .then((res) => res.json())
      .then((data: ModelInfo) => {
        setInfo(data)
        if (data.sampling) setSampling(data.sampling) // server's defaults
      })
      .catch(() => setInfo(null))
  }, [])
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Keep the newest message in view.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  function addMessage(role: Message['role'], text: string, error = false) {
    setMessages((prev) => [...prev, { id: nextId.current++, role, text, error }])
  }

  async function send(e: FormEvent) {
    e.preventDefault()
    const text = input.trim()
    if (!text || loading) return

    addMessage('user', text)
    setInput('')
    setLoading(true)
    const reply = await askModel(text, sampling)
    addMessage('assistant', reply.text, reply.error)
    setLoading(false)
    inputRef.current?.focus()
  }

  return (
    <div className="chat">
      <header className="chat-header">
        <h1>tiny-gpt</h1>
        <span className="chat-header-note">Step 4b · residual connection</span>
        {info && (
          <button
            type="button"
            className="params-toggle"
            onClick={() => setShowParams((s) => !s)}
            aria-expanded={showParams}
          >
            {fmt(info.total)} parameters {showParams ? '▴' : '▾'}
          </button>
        )}
      </header>

      {info && showParams && <ParametersPanel info={info} />}

      <main className="chat-messages" aria-live="polite">
        {messages.length === 0 && (
          <div className="chat-empty">
            <p>Type the start of a sentence and the model finishes it.</p>
            <p className="chat-empty-example">Try "The Dog" or "my mother"</p>
          </div>
        )}

        {messages.map((m) => (
          <div key={m.id} className={`message message-${m.role}${m.error ? ' message-error' : ''}`}>
            {m.text}
          </div>
        ))}

        {loading && <div className="message message-assistant message-typing">…</div>}
        <div ref={bottomRef} />
      </main>

      <SamplingControls value={sampling} onChange={setSampling} />

      <form className="chat-composer" onSubmit={send}>
        <input
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Message tiny-gpt"
          aria-label="Message"
          autoFocus
        />
        <button type="submit" disabled={loading || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  )
}

export default App
