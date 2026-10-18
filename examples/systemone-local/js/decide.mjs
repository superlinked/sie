// Typed decisions with the TypeSafe JS SDK, served by a local SIE.
//
// Ordinary TypeSafe SDK code: the client reads TYPESAFE_BASE_URL,
// TYPESAFE_DEFAULT_MODEL and TYPESAFE_API_KEY from the environment, and those
// point it at SIE (see ../README.md).
import { TypeSafeClient, choice, noul, score } from "@typesafe-ai/sdk";

const TICKET = "I was charged twice for my March invoice. Please refund one of the charges today.";

const client = new TypeSafeClient();
const { model, answers, usage } = await client.systemOne({
  state: TICKET,
  questions: {
    spam: noul("Is this message unsolicited spam?"),
    intent: choice("What does the customer want?", {
      billing: "payments, invoices or refunds",
      technical: "something is broken",
      other: null,
    }),
    urgency: score("How urgent is this request?", ["can wait", "this week", "today"]),
  },
});

console.log(`model: ${model}`);
console.log(`spam: P(yes) = ${answers.spam.noul.toFixed(2)}`);
console.log(`intent: ${answers.intent.choice} (confidence ${answers.intent.confidence.toFixed(2)})`);
console.log(`urgency: ${answers.urgency.score.toFixed(2)} of 0-2 (confidence ${answers.urgency.confidence.toFixed(2)})`);
console.log(`input tokens: ${usage.input_tokens}`);

const models = await client.models.list();
console.log(`models listed: ${models.length}`);
