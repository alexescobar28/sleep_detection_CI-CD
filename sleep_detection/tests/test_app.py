def test_reset_state():
    response = client.post('/reset')
    assert response.status_code == 200
    
def test_process_frame_no_image():
    response = client.post('/process_frame')
    assert response.status_code == 400