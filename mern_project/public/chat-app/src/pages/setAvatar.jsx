import React, {useState, useEffect} from 'react';
import {useNavigate} from 'react-router-dom';
import styled from 'styled-components';
import loader from '../assets/logo192.png';
import {ToastContainer, toast} from 'react-toastify';
import "react-toastify/dist/ReactToastify.css"
import { setAvatarRoute } from '../assets/utils/APIRoutes';
import axios from "axios";
import {Buffer} from 'buffer';

export default function SetAvatar() {
  const api = "https://api.multiavatar.com/45678945";
  const navigate = useNavigate();
  const [avatars, setAvatars] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedAvatar, setSelectedAvatar] = useState(undefined);
  const toastOptions = {
    position: "bottom-right",
    autoClose: 8000,
    pauseOnHover: true,
    draggable: true,
    theme: "dark",
  };
  const setProfilePicture = async () => {};
  const doSomething = async() =>{
    const data = [];
    for(let i = 0; i < 4; i++){
      const image = await axios.get(
        `${api}/${Math.round(Math.random() * 1000)}`
      );
      const buffer = new Buffer(image.data);
      data.push(buffer.toString("base64"));
    }
    setAvatars(data);
    setIsLoading(false);
  }
  useEffect(() => {
    doSomething();
  }, []);
  return (
    <>
      <Container>
        <div className="title-container">
          <h1>Pick an avatar as your profile picture</h1>
        </div>
        <div className="avatars">
          {avatars.map((avatar, index) => {
            return(
              <div 
                key = {index}
                className={`avatar ${
                  selectedAvatar === index ? "selected" : ""
                }`}
              >
                <img 
                  src={`data:image/svg+xml;base64,${avatar}`}
                  alt="avatar" 
                  onClick={()=>setSelectedAvatar(index)}
                />
              </div>
            )
          })}
        </div>
      </Container>
      <ToastContainer/>
    </>
  )
}
const Container = styled.div`
  display:flex;
  justify-content: center;
  align-items:center;
  flex-direction:column;
  gap:3rem;
  background-color: #131324;
`;
