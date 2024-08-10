const { z } = require('zod');
const { Tool } = require('@langchain/core/tools');
const { getEnvironmentVariable } = require('@langchain/core/utils/env');

const { AgentExecutor, createReactAgent } = require('langchain/agents');
const { pull } = require('langchain/hub');
const { OpenAI } = require('@langchain/openai');
const { PromptTemplate } = require('@langchain/core/prompts');
const { GoogleCustomSearch } = require('@langchain/community/tools/google_custom_search');

class ReActAgent extends Tool {
  static lc_name() {
    return 'ReActAgent';
  }

  constructor(fields = {}) {
    super(fields);
    this.envVarApiKey = 'GOOGLE_SEARCH_API_KEY';
    this.envVarSearchEngineId = 'GOOGLE_CSE_ID';
    this.envVarOpenAIApiKey = 'OPENAI_API_KEY';
    this.override = fields.override ?? false;
    this.apiKey = fields[this.envVarApiKey] ?? getEnvironmentVariable(this.envVarApiKey);
    this.searchEngineId =
      fields[this.envVarSearchEngineId] ?? getEnvironmentVariable(this.envVarSearchEngineId);
    this.openAIApiKey =
      fields[this.envVarOpenAIApiKey] ?? getEnvironmentVariable(this.envVarOpenAIApiKey);

    if (!this.override && (!this.apiKey || !this.searchEngineId || !this.openAIApiKey)) {
      throw new Error(
        `Missing ${this.envVarApiKey} or ${this.envVarSearchEngineId} or ${this.envVarOpenAIApiKey} environment variable.`,
      );
    }

    this.kwargs = fields?.kwargs ?? {};
    this.name = 'react_agent';
    this.description = 'A langchain based agent that will reason and act to answer the users query';

    this.schema = z.object({
      query: z.string().min(1).describe('The query string.'),
    });
  }

  async _call(input) {
    const validationResult = this.schema.safeParse(input);
    if (!validationResult.success) {
      throw new Error(`Validation failed: ${JSON.stringify(validationResult.error.issues)}`);
    }

    const { query } = validationResult.data;
    const prompt = (await pull) < PromptTemplate > 'hwchase17/react';
    const llm = new OpenAI({
      model: 'gpt-4o',
      temperature: 0,
      openAIApiKey: this.openAIApiKey,
    });
    const tools = [
      new GoogleCustomSearch({
        apiKey: this.apiKey,
        googleCSEId: this.searchEngineId,
      }),
    ];

    const agent = await createReactAgent({
      llm,
      tools,
      prompt,
    });
    const agentExecutor = new AgentExecutor({
      agent,
      tools,
    });
    const result = await agentExecutor.invoke({
      input: query,
    });
    return result.output;
  }
}

module.exports = ReActAgent;
