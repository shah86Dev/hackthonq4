import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug', '5ff'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config', '5ba'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content', 'a2b'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData', 'c3c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata', '156'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry', '88c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes', '000'),
    exact: true
  },
  {
    path: '/docs',
    component: ComponentCreator('/docs', '09b'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', '5e8'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', 'fe8'),
            routes: [
              {
                path: '/docs/',
                component: ComponentCreator('/docs/', '160'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/assessments/',
                component: ComponentCreator('/docs/assessments/', '79a'),
                exact: true
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
                path: '/docs/module1/chapter1',
                component: ComponentCreator('/docs/module1/chapter1', '7b0'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/module1/chapter2',
                component: ComponentCreator('/docs/module1/chapter2', '625'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/module1/chapter3',
                component: ComponentCreator('/docs/module1/chapter3', '764'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/module1/chapter4',
                component: ComponentCreator('/docs/module1/chapter4', '674'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/module2/chapter5',
                component: ComponentCreator('/docs/module2/chapter5', 'aba'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/module2/chapter6',
                component: ComponentCreator('/docs/module2/chapter6', 'c01'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/module2/chapter7',
                component: ComponentCreator('/docs/module2/chapter7', '1e3'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/module2/chapter8',
                component: ComponentCreator('/docs/module2/chapter8', '241'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/module3/chapter10',
                component: ComponentCreator('/docs/module3/chapter10', '11a'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/module3/chapter11',
                component: ComponentCreator('/docs/module3/chapter11', 'fff'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/module3/chapter12',
                component: ComponentCreator('/docs/module3/chapter12', '9ca'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/module3/chapter9',
                component: ComponentCreator('/docs/module3/chapter9', 'c4b'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/module4/chapter13',
                component: ComponentCreator('/docs/module4/chapter13', 'ebe'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/module4/chapter14',
                component: ComponentCreator('/docs/module4/chapter14', 'f7a'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/module4/chapter15',
                component: ComponentCreator('/docs/module4/chapter15', '7cc'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/docs/module4/chapter16',
                component: ComponentCreator('/docs/module4/chapter16', '8b0'),
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
    path: '*',
    component: ComponentCreator('*'),
  },
];
