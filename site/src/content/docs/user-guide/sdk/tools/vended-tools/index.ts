import { Agent } from '@strands-agents/sdk'
import { bash } from '@strands-agents/sdk/vended-tools/bash'
import { fileEditor } from '@strands-agents/sdk/vended-tools/file-editor'
import { httpRequest } from '@strands-agents/sdk/vended-tools/http-request'
import { notebook } from '@strands-agents/sdk/vended-tools/notebook'

// Agent with vended tools example
async function agentWithVendedToolsExample() {
  // --8<-- [start:agent_with_vended_tools]
  const agent = new Agent({
    tools: [bash, fileEditor, httpRequest, notebook],
  })
  // --8<-- [end:agent_with_vended_tools]
}

// Combined tools example - development workflow
async function combinedToolsExample() {
  // --8<-- [start:combined_tools_example]
  const agent = new Agent({
    tools: [bash, fileEditor, notebook],
    systemPrompt: [
      'You are a software development assistant.',
      'When given a feature to implement:',
      '1. Use the notebook tool to create a plan with a checklist of steps',
      '2. Work through each step, checking them off as you go',
      '3. Use the bash tool to run tests and verify your changes',
    ].join('\n'),
  })

  // Agent plans the work, implements it, and tracks progress
  await agent.invoke(
    'Add input validation to the createUser function in src/users.ts. ' +
      'It should reject empty names and invalid email formats.'
  )
  // --8<-- [end:combined_tools_example]
}
