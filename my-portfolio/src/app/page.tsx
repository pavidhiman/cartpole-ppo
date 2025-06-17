export default function Home() {
  return (
    <main className="max-w-3xl mx-auto p-6">
      <h1 className="text-4xl font-bold mb-8">Pavi Dhiman</h1>

      {/* 2025 */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">2025</h2>
        <ul className="list-disc list-inside space-y-2">
          <li>
            <strong>AI Interview Copilot</strong> – A conversational assistant for hiring managers.{' '}
            <a href="#" className="underline">GitHub</a>
          </li>
          {/* …add more 2025 projects here */}
        </ul>
      </section>

      {/* 2024 */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">2024</h2>
        <ul className="list-disc list-inside space-y-2">
          <li>
            <strong>Sips Don’t Lie</strong> – Smart water bottle that shames you into hydrating.{' '}
            <a href="#" className="underline">Devpost</a>
          </li>
          {/* …add more 2024 projects here */}
        </ul>
      </section>

      {/* About or other years */}
      {/* <section>…</section> */}
    </main>
  );
}
