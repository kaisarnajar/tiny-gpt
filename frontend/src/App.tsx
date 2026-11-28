import { useEffect, useRef, useState, type FormEvent } from 'react'
import './App.css'

type Message = { role: 'user' | 'assistant'; text: string; error?: boolean }

const EXAMPLES = ['Once upon a time', 'The water cycle is', 'def is_prime(n):']

export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const messagesRef = useRef<HTMLElement>(null)
  const [total, setTotal] = useState(0)
  const [temperature, setTemperature] = useState(1.0)
  const [topK, setTopK] = useState(0)
  const [topP, setTopP] = useState(0.9)

  useEffect(() => {
    fetch('/api/info')
      .then((res) => res.json())
      .then((data) => setTotal(data.parameters))
  }, [])

  useEffect(() => {
    messagesRef.current?.scrollTo(0, messagesRef.current.scrollHeight)
  }, [messages])

  async function send(e: FormEvent) {
    e.preventDefault()
    const prompt = input.trim()
    if (!prompt || loading) return

    setMessages((m) => [...m, { role: 'user', text: prompt }])
    setInput('')
    setLoading(true)
    try {
      const res = await fetch('/api/complete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, temperature, top_k: topK, top_p: topP }),
      })
      if (!res.ok) {
        const data = await res.json()
        setMessages((m) => [...m, { role: 'assistant', text: data.error, error: true }])
      } else {
        setMessages((m) => [...m, { role: 'assistant', text: prompt }])
        const reader = res.body!.pipeThrough(new TextDecoderStream()).getReader()
        let buffer = ''
        while (true) {
          const { value, done } = await reader.read()
          if (done) break
          buffer += value
          const events = buffer.split('\n\n')
          buffer = events.pop()!
          const text = events.map((event) => JSON.parse(event.slice('data: '.length)).text).join('')
          setMessages((m) => [...m.slice(0, -1), { role: 'assistant', text: m[m.length - 1].text + text }])
        }
      }
    } catch {
      setMessages((m) => [...m, { role: 'assistant', text: 'Could not reach the server.', error: true }])
    }
    setLoading(false)
  }

  const greedy = temperature === 0

  return (
    <div className="chat">
      <header className="chat-header">
        <h1>tiny-gpt</h1>
        <span className="params-count">{total.toLocaleString('en-US')} parameters</span>
      </header>

      <main className="chat-messages" ref={messagesRef}>
        {messages.length === 0 && (
          <div className="chat-empty">
            <p>Type the start of a story, an article or some Python code, and the model continues it.</p>
            <div className="chat-examples">
              {EXAMPLES.map((example) => (
                <button key={example} type="button" onClick={() => setInput(example)}>
                  {example}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`message message-${m.role}${m.error ? ' message-error' : ''}`}>
            {m.text}
          </div>
        ))}
        {loading && messages[messages.length - 1].role === 'user' && (
          <div className="message message-assistant message-typing">…</div>
        )}
      </main>

      <section className="sampling">
        <label>
          <span>Temperature</span>
          <input type="range" min={0} max={2} step={0.1} value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))} />
          <output>{greedy ? '0 · greedy' : temperature.toFixed(1)}</output>
        </label>
        <label className={greedy ? 'off' : ''}>
          <span>Top-k</span>
          <input type="range" min={0} max={20} step={1} value={topK} disabled={greedy}
            onChange={(e) => setTopK(Number(e.target.value))} />
          <output>{topK === 0 ? 'off' : topK}</output>
        </label>
        <label className={greedy ? 'off' : ''}>
          <span>Top-p</span>
          <input type="range" min={0.05} max={1} step={0.05} value={topP} disabled={greedy}
            onChange={(e) => setTopP(Number(e.target.value))} />
          <output>{topP >= 1 ? 'off' : topP.toFixed(2)}</output>
        </label>
      </section>

      <form className="chat-composer" onSubmit={send}>
        <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Message tiny-gpt" autoFocus />
        <button type="submit" disabled={loading || !input.trim()}>Send</button>
      </form>
    </div>
  )
}
