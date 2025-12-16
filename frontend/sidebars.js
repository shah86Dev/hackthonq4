// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  textbookSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      items: ['intro', 'setup'],
    },
    {
      type: 'category',
      label: 'Module 1: ROS2',
      items: [
        'module1/chapter1',
        'module1/chapter2',
        'module1/chapter3',
        'module1/chapter4'
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Gazebo & Unity',
      items: [
        'module2/chapter5',
        'module2/chapter6',
        'module2/chapter7',
        'module2/chapter8'
      ],
    },
    {
      type: 'category',
      label: 'Module 3: NVIDIA Isaac',
      items: [
        'module3/chapter9',
        'module3/chapter10',
        'module3/chapter11',
        'module3/chapter12'
      ],
    },
    {
      type: 'category',
      label: 'Module 4: VLA Models',
      items: [
        'module4/chapter13',
        'module4/chapter14',
        'module4/chapter15',
        'module4/chapter16'
      ],
    },
    {
      type: 'category',
      label: 'Resources',
      items: [
        'glossary',
        'references',
        'instructor-guide',
        'lab-manual'
      ],
    }
  ],
};

export default sidebars;