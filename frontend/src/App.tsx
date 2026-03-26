import React from 'react'
import { Layout, Menu } from 'antd'
import ModelManagement from './components/ModelManagement'
import AudioProcessor from './components/AudioProcessor'
import RealtimeProcessor from './components/RealtimeProcessor'
import ResultManager from './components/ResultManager'

const { Header, Sider, Content } = Layout

function App() {
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ background: '#001529', padding: 0 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0 24px' }}>
          <div style={{ color: '#fff', fontSize: 18px, fontWeight: bold }}>
            CantioAI Web Interface
          </div>
          <div>
            <span style={{ color: '#fff', marginRight: 16px }}>User</span>
            <a href="#" style={{ color: '#fff', marginRight: 16px }}>Settings</a>
            <a href="#" style={{ color: '#fff' }}>Help</a>
          </div>
        </div>
      </Header>
      <Layout>
        <Sider width={200} style={{ background: '#002140' }}>
          <Menu theme="dark" mode="inline" defaultSelectedKeys={['1']}>
            <Menu.Item key="1" icon={<ModelManagement />}>
              Model Management
            </Menu.Item>
            <Menu.Item key="2" icon={<AudioProcessor />}>
              Audio Processing
            </Menu.Item>
            <Menu.Item key="3" icon={<RealtimeProcessor />}>
              Real-time Processing
            </Menu.Item>
            <Menu.Item key="4" icon={<ResultManager />}>
              Results & History
            </Menu.Item>
          </Menu>
        </Sider>
        <Layout style={{ padding: '0 24px 24px' }}>
          <Content style={{ margin: '24px 16px 0', padding: 24, background: '#fff', minHeight: 280 }}>
            <div>
              <h2>Welcome to CantioAI Web Interface</h2>
              <p>
                This is the main dashboard for the CantioAI voice conversion system.
                Use the sidebar to navigate between different modules:
              </p>
              <ul>
                <li><strong>Model Management</strong>: Load, unload, and manage AI models</li>
                <li><strong>Audio Processing</strong>: Upload, process, and download audio files</li>
                <li><strong>Real-time Processing</strong>: Process audio from microphone in real-time</li>
                <li><strong>Results & History</strong>: View processing history and manage results</li>
              </ul>
            </div>
          </Content>
        </Layout>
      </Layout>
    </Layout>
  )
}

export default App