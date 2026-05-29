import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import type {Options as DocsOptions} from '@docusaurus/plugin-content-docs';
import type {Options as RedirectOptions} from '@docusaurus/plugin-client-redirects';

const BASE_URL: string = process.env.DOCS_BASE_URL ?? '/';
const SITE_URL: string = BASE_URL === '/mellea/'
  ? 'https://planetf1.github.io'
  : 'https://docs.mellea.ai';

const config: Config = {
  title: 'Mellea',
  tagline: 'Build predictable AI without guesswork',
  url: SITE_URL,
  baseUrl: BASE_URL,

  onBrokenLinks: 'warn',
  onBrokenAnchors: 'warn',
  onDuplicateRoutes: 'throw',

  favicon: 'images/favicon.svg',
  trailingSlash: false,
  i18n: {defaultLocale: 'en', locales: ['en']},
  markdown: {format: 'detect'},

  presets: [
    [
      'classic',
      {
        docs: {
          path: 'docs',
          routeBasePath: '/',
          sidebarPath: './sidebars.ts',
          showLastUpdateAuthor: true,
          showLastUpdateTime: true,
          editUrl: 'https://github.com/generative-computing/mellea/edit/main/',
        } satisfies DocsOptions,
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
        sitemap: {changefreq: 'weekly', priority: 0.5},
      } satisfies Preset.Options,
    ],
  ],

  plugins: [
    [
      '@docusaurus/plugin-client-redirects',
      {
        redirects: [
          {from: '/guide/glossary', to: '/reference/glossary'},
          {from: '/overview/overview', to: '/getting-started/quickstart'},
          {from: '/overview/mellea-welcome', to: '/'},
          {from: '/overview/quick-start', to: '/getting-started/quickstart'},
          {from: '/overview/project-mellea', to: '/concepts/generative-programming'},
          {from: '/overview/generative-programming', to: '/concepts/generative-programming'},
          {from: '/overview/architecture', to: '/how-to/backends-and-configuration'},
          {from: '/core-concept/instruct-validate-repair', to: '/concepts/instruct-validate-repair'},
          {from: '/core-concept/requirements', to: '/concepts/requirements-system'},
          {from: '/core-concept/generative-slots', to: '/how-to/generative-functions'},
          {from: '/core-concept/mobjects', to: '/concepts/mobjects-and-mify'},
          {from: '/core-concept/agents', to: '/how-to/tools-and-agents'},
          {from: '/core-concept/context-management', to: '/how-to/use-context-and-sessions'},
          {from: '/core-concept/alora', to: '/advanced/lora-and-alora-adapters'},
          {from: '/core-concept/tuning', to: '/advanced/lora-and-alora-adapters'},
          {from: '/core-concept/modeloptions', to: '/how-to/configure-model-options'},
          {from: '/core-concept/interoperability', to: '/integrations/mcp'},
          {from: '/integrations/mcp-and-m-serve', to: '/integrations/mcp'},
          {from: '/core-concept/adapters', to: '/how-to/tools-and-agents'},
          {from: '/core-concept/contribution-guide', to: '/community/contributing-guide'},
          {from: '/core-concept/prompt-engineering', to: '/advanced/mellea-core-internals'},
          {from: '/integrations/bedrock-and-watsonx', to: '/integrations/bedrock'},
          {from: '/integrations/huggingface-and-vllm', to: '/integrations/huggingface'},
          {from: '/integrations/langchain-and-smolagents', to: '/integrations/langchain'},
          {from: '/dev/constrained-decoding', to: '/advanced/mellea-core-internals'},
          {from: '/dev/generate-ctx-signature', to: '/advanced/mellea-core-internals'},
          {from: '/dev/intrinsics-and-adapters', to: '/advanced/intrinsics'},
          {from: '/dev/mellea-library', to: '/concepts/generative-programming'},
          {from: '/dev/mify', to: '/concepts/mobjects-and-mify'},
          {from: '/dev/requirement-alora-rerouting', to: '/advanced/lora-and-alora-adapters'},
          {from: '/dev/spans', to: '/observability/tracing'},
          {from: '/dev/tool-calling', to: '/how-to/tools-and-agents'},
          {from: '/api/cli/m', to: '/reference/cli'},
          {from: '/api/cli/alora/commands', to: '/reference/cli'},
          {from: '/api/cli/alora/intrinsic_uploader', to: '/reference/cli'},
          {from: '/api/cli/alora/readme_generator', to: '/reference/cli'},
          {from: '/api/cli/alora/train', to: '/reference/cli'},
          {from: '/api/cli/alora/upload', to: '/reference/cli'},
          {from: '/api/cli/decompose/decompose', to: '/reference/cli'},
          {from: '/api/cli/decompose/pipeline', to: '/reference/cli'},
          {from: '/api/cli/decompose/utils', to: '/reference/cli'},
          {from: '/api/cli/eval/commands', to: '/reference/cli'},
          {from: '/api/cli/eval/eval', to: '/reference/cli'},
          {from: '/api/cli/eval/runner', to: '/reference/cli'},
          {from: '/api/cli/serve/app', to: '/reference/cli'},
          {from: '/api/cli/serve/models', to: '/reference/cli'},
          {from: '/api/cli/fix/commands', to: '/reference/cli'},
          {from: '/api/cli/fix/async_fixer', to: '/reference/cli'},
          {from: '/api/cli/fix/genstub_fixer', to: '/reference/cli'},
          {from: '/guide/generative-functions', to: '/how-to/generative-functions'},
          {from: '/guide/tools-and-agents', to: '/how-to/tools-and-agents'},
          {from: '/guide/working-with-data', to: '/how-to/working-with-data'},
          {from: '/guide/backends-and-configuration', to: '/how-to/backends-and-configuration'},
          {from: '/evaluation-and-observability/evaluate-with-llm-as-a-judge', to: '/how-to/evaluate-with-llm-as-a-judge'},
          {from: '/evaluation-and-observability/telemetry', to: '/observability/telemetry'},
          {from: '/evaluation-and-observability/tracing', to: '/observability/tracing'},
          {from: '/evaluation-and-observability/metrics', to: '/observability/metrics'},
          {from: '/evaluation-and-observability/logging', to: '/observability/logging'},
          {from: '/guide/act-and-aact', to: '/how-to/act-and-aact'},
          {from: '/guide/m-decompose', to: '/how-to/m-decompose'},
          {from: '/advanced/security-and-taint-tracking', to: '/how-to/safety-guardrails'},
        ],
      } satisfies RedirectOptions,
    ],
    [
      '@easyops-cn/docusaurus-search-local',
      {
        hashed: true,
        indexBlog: false,
        indexPages: false,
        docsRouteBasePath: ['/'],
        language: ['en'],
        explicitSearchResultPath: true,
      },
    ],
  ],

  themeConfig: {
    image: 'images/mellea_draft_logo_300.png',
    metadata: [
      {name: 'keywords', content: 'mellea, generative programming, AI, LLM, Python'},
      {name: 'description', content: 'Build predictable AI without guesswork'},
    ],
    navbar: {
      title: 'Mellea',
      logo: {
        alt: 'Mellea',
        src: 'logo/logo-light.svg',
        srcDark: 'logo/logo-dark.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          type: 'docSidebar',
          sidebarId: 'apiSidebar',
          position: 'left',
          label: 'API Reference',
        },
        {
          href: 'https://generative-computing.github.io/blog/',
          label: 'Blog',
          position: 'right',
        },
        {
          href: 'https://github.com/generative-computing/mellea/discussions',
          label: 'Community',
          position: 'right',
        },
        {
          href: 'https://github.com/generative-computing/mellea/blob/main/CONTRIBUTING.md',
          label: 'Contribution Guide',
          position: 'right',
        },
        {
          href: 'https://github.com/generative-computing/mellea/issues',
          label: 'Support',
          position: 'right',
        },
        {
          href: 'https://github.com/generative-computing/mellea',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          items: [
            {label: 'Blog', href: 'https://generative-computing.github.io/blog/'},
            {label: 'Community', href: 'https://github.com/generative-computing/mellea/discussions'},
            {label: 'Contribution Guide', href: 'https://github.com/generative-computing/mellea/blob/main/CONTRIBUTING.md'},
            {label: 'Support', href: 'https://github.com/generative-computing/mellea/issues'},
            {label: 'GitHub', href: 'https://github.com/generative-computing/mellea'},
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Mellea Contributors.`,
    },
    colorMode: {
      respectPrefersColorScheme: true,
    },
    tableOfContents: {
      minHeadingLevel: 2,
      maxHeadingLevel: 4,
    },
    prism: {
      additionalLanguages: ['python', 'bash', 'yaml', 'json'],
    },
  } satisfies Preset.ThemeConfig,

  scripts: [
    {src: `${BASE_URL}analytics.js`, async: true},
  ],
};

export default config;
