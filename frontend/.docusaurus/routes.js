import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/chat',
    component: ComponentCreator('/chat', '07b'),
    exact: true
  },
  {
    path: '/docs',
    component: ComponentCreator('/docs', '8f8'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', '07c'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', '25d'),
            routes: [
              {
                path: '/docs/assessments/',
                component: ComponentCreator('/docs/assessments/', '79a'),
                exact: true
              },
              {
                path: '/docs/chatbot/introduction',
                component: ComponentCreator('/docs/chatbot/introduction', '142'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/glossary',
                component: ComponentCreator('/docs/glossary', '2d4'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/instructor-guide',
                component: ComponentCreator('/docs/instructor-guide', 'd08'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/intro',
                component: ComponentCreator('/docs/intro', '853'),
                exact: true
              },
              {
                path: '/docs/lab-manual',
                component: ComponentCreator('/docs/lab-manual', '4e8'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/labs/lab-manual',
                component: ComponentCreator('/docs/labs/lab-manual', '4a8'),
                exact: true
              },
              {
                path: '/docs/module1/1-embodied-cognition',
                component: ComponentCreator('/docs/module1/1-embodied-cognition', '350'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/module1/1-introduction-to-physical-ai',
                component: ComponentCreator('/docs/module1/1-introduction-to-physical-ai', '42b'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/module1/1-learning-in-physical-systems',
                component: ComponentCreator('/docs/module1/1-learning-in-physical-systems', '962'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/module1/1-sensorimotor-coordination',
                component: ComponentCreator('/docs/module1/1-sensorimotor-coordination', 'bf9'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/module2/2-human-robot-interaction',
                component: ComponentCreator('/docs/module2/2-human-robot-interaction', '663'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/module2/2-humanoid-robot-platforms',
                component: ComponentCreator('/docs/module2/2-humanoid-robot-platforms', '535'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/module2/2-locomotion-and-gait-control',
                component: ComponentCreator('/docs/module2/2-locomotion-and-gait-control', '771'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/module2/2-manipulation-and-grasping',
                component: ComponentCreator('/docs/module2/2-manipulation-and-grasping', '4bb'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/module3/3-embodied-learning',
                component: ComponentCreator('/docs/module3/3-embodied-learning', '9a1'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/module3/3-multi-robot-physical-ai',
                component: ComponentCreator('/docs/module3/3-multi-robot-physical-ai', 'aba'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/module3/3-physics-informed-neural-networks',
                component: ComponentCreator('/docs/module3/3-physics-informed-neural-networks', '34d'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/module3/3-sim-to-real-transfer',
                component: ComponentCreator('/docs/module3/3-sim-to-real-transfer', '9f0'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/module4/4-future-directions',
                component: ComponentCreator('/docs/module4/4-future-directions', 'c07'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/module4/4-industrial-physical-ai',
                component: ComponentCreator('/docs/module4/4-industrial-physical-ai', '22a'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/module4/4-safety-and-ethics-in-physical-ai',
                component: ComponentCreator('/docs/module4/4-safety-and-ethics-in-physical-ai', '279'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/module4/4-service-robotics',
                component: ComponentCreator('/docs/module4/4-service-robotics', 'a95'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/references',
                component: ComponentCreator('/docs/references', '044'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/test',
                component: ComponentCreator('/docs/test', 'cf7'),
                exact: true
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/',
    component: ComponentCreator('/', '2e1'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
