import numpy as np

# Define keyframes for a 60-second aggressive race quad flight.
# Columns: t_s, wx_cmd_radps (roll), wy_cmd_radps (pitch), wz_cmd_radps (yaw), thr_frac (throttle)

keyframes = [
    [0.0, 0.0, 0.0, 0.0, 0.25],
    [1.0, 0.0, 0.0, 0.0, 0.25],   # Hover
    
    # 1. Accelerate forward
    [1.5, 0.0, 2.0, 0.0, 0.4],    # Pitch down, increase throttle
    [4.0, 0.0, 2.0, 0.0, 0.4],
    
    # 2. Hard braking
    [4.5, 0.0, -3.0, 0.0, 0.1],   # Pitch up hard, cut throttle to prevent ballooning
    [5.5, 0.0, -3.0, 0.0, 0.1],
    [6.0, 0.0, 0.0, 0.0, 0.25],   # Level out
    
    # 3. Snap roll right
    [7.8, 0.0, 0.0, 0.0, 0.25],
    [7.9, 0.0, 0.0, 0.0, 0.5],    # Throttle punch before roll
    [8.1, 10.0, 0.0, 0.0, 0.05],  # Aggressive roll right, cut throttle during rotation
    [8.5, 10.0, 0.0, 0.0, 0.05],
    [8.6, 0.0, 0.0, 0.0, 0.4],    # Stop roll, catch with throttle
    [9.0, 0.0, 0.0, 0.0, 0.25],
    
    # 4. Sharp 90-degree yaw turn
    [10.0, 0.0, 0.0, 0.0, 0.25],
    [10.2, 0.0, 0.0, 5.0, 0.35],  # Yaw left, slight throttle bump
    [10.7, 0.0, 0.0, 5.0, 0.35],
    [10.9, 0.0, 0.0, 0.0, 0.25],
    
    # 5. Split-S Maneuver (Half roll inverted, then half loop down)
    [13.0, 0.0, 0.0, 0.0, 0.25],
    [13.1, 0.0, 0.0, 0.0, 0.4],   # Small pop
    [13.2, 8.0, 0.0, 0.0, 0.05],  # Roll 180 degrees inverted, cut throttle
    [13.6, 8.0, 0.0, 0.0, 0.05],
    [13.8, 0.0, 0.0, 0.0, 0.05],  # Inverted
    [14.5, 0.0, -4.0, 0.0, 0.3],  # Pull pitch to complete half loop, add throttle
    [15.5, 0.0, -4.0, 0.0, 0.4],  # Pull out
    [16.0, 0.0, 0.0, 0.0, 0.25],  # Level out
    
    # 6. Power Loop (Full backward flip)
    [19.5, 0.0, 0.0, 0.0, 0.25],
    [19.8, 0.0, 0.0, 0.0, 0.7],   # Big punch out
    [20.0, 0.0, 0.0, 0.0, 0.05],  # Cut throttle
    [20.5, 0.0, -6.0, 0.0, 0.05], # Pull back hard
    [21.6, 0.0, -6.0, 0.0, 0.1],  # Over the top
    [21.9, 0.0, 0.0, 0.0, 0.6],   # Catch it
    [22.5, 0.0, 0.0, 0.0, 0.25],  # Loop complete
    
    # 7. Slalom sequence (weaving)
    [25.0, 0.0, 0.0, 0.0, 0.25],
    [26.0, 3.0, 0.0, 0.0, 0.35],  # Bank right, hold higher throttle
    [27.0, -3.0, 0.0, 0.0, 0.35], # Bank left
    [28.0, 3.0, 0.0, 0.0, 0.35],  # Bank right
    [29.0, -3.0, 0.0, 0.0, 0.35], # Bank left
    [30.0, 0.0, 0.0, 0.0, 0.25],  # Level out
    
    # 8. High-speed front flip
    [34.8, 0.0, 0.0, 0.0, 0.25],
    [35.0, 0.0, 0.0, 0.0, 0.6],   # Punch out
    [35.2, 0.0, 10.0, 0.0, 0.05], # Pitch forward fast, cut throttle
    [35.8, 0.0, 10.0, 0.0, 0.05],
    [36.0, 0.0, 0.0, 0.0, 0.5],   # Catch
    [36.5, 0.0, 0.0, 0.0, 0.25],
    
    # 9. Yaw spin (freestyle 360/720)
    [40.0, 0.0, 0.0, 0.0, 0.25],
    [40.5, 0.0, 0.0, -8.0, 0.3],  # Fast yaw right
    [41.5, 0.0, 0.0, -8.0, 0.3],
    [42.0, 0.0, 0.0, 0.0, 0.25],
    
    # 10. Coasting / Return
    [45.0, 0.0, 0.0, 0.0, 0.25],
    [46.0, 0.0, -1.5, 0.0, 0.15], # Gentle braking
    [49.0, 0.0, -1.5, 0.0, 0.15],
    [50.0, 0.0, 0.0, 0.0, 0.25],  # Hovering
    
    # 11. Final adjustments for landing
    [55.0, 0.0, 0.0, 0.0, 0.25],
    [58.0, 0.0, 0.0, 0.0, 0.15],  # Descend
    [60.0, 0.0, 0.0, 0.0, 0.0]    # Landed (motors off)
]

np.savetxt('guidance.csv', keyframes, delimiter=',', header='t_s,wx_cmd_radps,wy_cmd_radps,wz_cmd_radps,thr_frac', comments='', fmt='%.3f')
print("guidance.csv generated successfully for a 60-second flight with variable throttle.")