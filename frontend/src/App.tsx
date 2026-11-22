import { useEffect, useRef, useState, type FormEvent } from 'react'
import './App.css'

type Message = {
  id: number
  role: 'user' | 'assistant'
  text: string
  error?: boolean
}

type CompleteResponse = { sentence: string } | { error: string }

type ModelInfo = {
  config: { vocab_size: number; d_model: number; context_len: number }
  layers: { name: string; shape: string; formula: string; calculation: string; params: number }[]
  total: number
  equation: string
}

const fmt = (n: number) => n.toLocaleString('en-US')

function ParametersPanel({ info }: { info: ModelInfo }) {
  const { vocab_size, d_model, context_len } = info.config
  return (
    <section className="params" aria-label="Parameter calculation">
      <p className="params-config">
        vocab_size = {vocab_size} · d_model = {d_model} · context_len = {context_len}
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

// Only the latest message goes to the backend; the history stays in the browser.
async function askModel(message: string): Promise<{ text: string; error?: boolean }> {
  try {
    const res = await fetch('/api/complete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: message }),
    })
    const data: CompleteResponse = await res.json()
    return 'error' in data ? { text: data.error, error: true } : { text: data.sentence }
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
  const nextId = useRef(0)

  // Model size + parameter calculation, fetched once when the page opens.
  useEffect(() => {
    fetch('/api/info')
      .then((res) => res.json())
      .then(setInfo)
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
    const reply = await askModel(text)
    addMessage('assistant', reply.text, reply.error)
    setLoading(false)
    inputRef.current?.focus()
  }

  return (
    <div className="chat">
      <header className="chat-header">
        <h1>tiny-gpt</h1>
        <span className="chat-header-note">Step 2 · embeddings</span>
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
