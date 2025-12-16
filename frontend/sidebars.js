// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  textbookSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      items: ['intro'],
    },
    {
      type: 'category',
      label: 'Module 1: Foundations of Physical AI',
      items: [
        'module1/1-introduction-to-physical-ai',
        'module1/1-sensorimotor-coordination',
        'module1/1-embodied-cognition',
        'module1/1-learning-in-physical-systems'
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Humanoid Robotics Fundamentals',
      items: [
        'module2/2-humanoid-robot-platforms',
        'module2/2-locomotion-and-gait-control',
        'module2/2-manipulation-and-grasping',
        'module2/2-human-robot-interaction'
      ],
    },
    {
      type: 'category',
      label: 'Module 3: Advanced Physical AI Techniques',
      items: [
        'module3/3-sim-to-real-transfer',
        'module3/3-physics-informed-neural-networks',
        'module3/3-embodied-learning',
        'module3/3-multi-robot-physical-ai'
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Real-World Applications',
      items: [
        'module4/4-industrial-physical-ai',
        'module4/4-service-robotics',
        'module4/4-safety-and-ethics-in-physical-ai',
        'module4/4-future-directions'
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